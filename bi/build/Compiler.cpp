/**
 * @file
 */
#include "Compiler.hpp"

#include "bi/build/misc.hpp"
#include "bi/visitor/Typer.hpp"
#include "bi/visitor/Resolver.hpp"
#include "bi/io/bi_ostream.hpp"
#include "bi/io/cpp_ostream.hpp"
#include "bi/io/hpp_ostream.hpp"
#include "bi/lexer.hpp"

#include "boost/filesystem/fstream.hpp"

#include <getopt.h>
#include <dlfcn.h>
#include <cstdlib>

namespace fs = boost::filesystem;

bi::Compiler* compiler = nullptr;
std::stringstream raw;

bi::Compiler::Compiler() :
    enable_std(false),
    enable_import(false) {
  //
}

bi::Compiler::Compiler(int argc, char** argv) :
    enable_std(true),
    enable_import(true) {
  enum {
    INCLUDE_DIR_ARG = 256,
    LIB_DIR_ARG,
    ENABLE_STD_ARG,
    DISABLE_STD_ARG
  };

  /* command-line arguments */
  int c, option_index;
  option long_options[] = {
      { "include-dir", required_argument, 0, INCLUDE_DIR_ARG },
      { "lib-dir", required_argument, 0, LIB_DIR_ARG },
      { "enable-std", no_argument, 0, ENABLE_STD_ARG },
      { "disable-std", no_argument, 0, DISABLE_STD_ARG },
      { 0, 0, 0, 0 } };
  const char* short_options = "o:D:I:L:";

  opterr = 0; // handle error reporting ourselves
  c = getopt_long(argc, argv, short_options, long_options, &option_index);
  while (c != -1) {
    switch (c) {
    case INCLUDE_DIR_ARG:
      include_dirs.push_back(optarg);
      break;
    case LIB_DIR_ARG:
      lib_dirs.push_back(optarg);
      break;
    case ENABLE_STD_ARG:
      enable_std = true;
      break;
    case DISABLE_STD_ARG:
      enable_std = false;
      break;
    case 'o':
      output_file = optarg;
      break;
    case 'D':
      // ignore
      break;
    case 'I':
      include_dirs.push_back(optarg);
      break;
    case 'L':
      lib_dirs.push_back(optarg);
      break;
    default:
      // ignore anything else, don't error, as this allows the program to
      // take any options that might be given to some other compiler through
      // a Makefile.
      break;
    }
    c = getopt_long(argc, argv, short_options, long_options, &option_index);
  }

  /* remaining arguments are input files */
  for (; optind < argc; ++optind) {
    fs::path path(argv[optind]);
    input_files.push_back(path.string());
  }

  /* name of standard library */
  if (enable_std) {
    Path path(new Name("standard"));
    standard = find(include_dirs, path.file()).string();
  }
}

bi::Compiler::~Compiler() {
  //
}

void bi::Compiler::parse() {
  /* queue standard library */
  if (enable_std) {
    queue(standard, false);
  }

  /* queue input files */
  for (auto iter = input_files.begin(); iter != input_files.end(); ++iter) {
    queue(*iter, enable_std);
  }

  /* parse all input files, and any imported files along the way */
  while (!unparsed.empty()) {
    auto name = *unparsed.begin();
    parse(name);
    parsed.insert(name);
    unparsed.erase(name);
  }
}

void bi::Compiler::resolve() {
  for (auto file : files) {
    Typer typer;
    file->accept(&typer);
  }
  for (auto file : files) {
    Resolver resolver;
    file->accept(&resolver);
  }
}

void bi::Compiler::gen() {
  for (auto iter = input_files.begin(); iter != input_files.end(); ++iter) {
    gen(*iter);
  }
}

void bi::Compiler::setRoot(Statement* root) {
  if (std) {
    Import* import = new Import(new Path(new Name("standard")),
        filesByName[standard]);
    file->root = new List<Statement>(import, root);
  } else {
    file->root = root;
  }
}

bi::File* bi::Compiler::import(const Path* path) {
  if (enable_import) {
    return queue(find(include_dirs, path->file()).string(), std);
  } else {
    return nullptr;
  }
}

bi::File* bi::Compiler::queue(const std::string name, const bool std) {
  if (filesByName.find(name) == filesByName.end()) {
    files.push_back(new File(name));
    filesByName.insert(std::make_pair(name, files.back()));
    stds.insert(std::make_pair(name, std));
    unparsed.insert(name);
  }
  return filesByName[name];
}

void bi::Compiler::parse(const std::string name) {
  /* pre-condition */
  assert(unparsed.find(name) != unparsed.end());

  yyin = fopen(name.c_str(), "r");
  if (!yyin) {
    throw FileNotFoundException(name.c_str());
  }

  file = filesByName[name];  // member variable needed by GNU Bison parser
  std = stds[name];
  yyreset();
  do {
    try {
      yyparse();
    } catch (bi::Exception& e) {
      yyerror(e.msg.c_str());
    }
  } while (!feof(yyin));
  fclose(yyin);
}

void bi::Compiler::gen(const std::string name) {
  /* pre-condition */
  assert(parsed.find(name) != parsed.end());

  fs::path hppPath, cppPath;
  if (!output_file.empty()) {
    cppPath = output_file;
    hppPath = output_file;
  } else {
    cppPath = name;
    hppPath = name;
  }
  hppPath.replace_extension(".hpp");
  cppPath.replace_extension(".cpp");

  fs::ofstream hppStream(hppPath);
  fs::ofstream cppStream(cppPath);

  hpp_ostream hppOutput(hppStream);
  cpp_ostream cppOutput(cppStream);

  setStates(File::UNGENERATED);
  hppOutput << filesByName[name];
  setStates(File::UNGENERATED);
  cppOutput << filesByName[name];
}

void bi::Compiler::setStates(const File::State state) {
  for (auto file : files) {
    file->state = state;
  }
}
