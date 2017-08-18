/**
 * @file
 */
#include "Compiler.hpp"

#include "bi/build/misc.hpp"
#include "bi/visitor/Typer.hpp"
#include "bi/visitor/Importer.hpp"
#include "bi/visitor/ResolverHeader.hpp"
#include "bi/visitor/ResolverSource.hpp"
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

bi::Compiler::Compiler(const std::list<fs::path>& include_dirs,
    const std::list<fs::path> lib_dirs, const bool enable_std) :
    include_dirs(include_dirs),
    lib_dirs(lib_dirs),
    enable_std(enable_std) {
  //
}

bi::Compiler::Compiler(int argc, char** argv) :
    enable_std(true) {
  enum {
    INCLUDE_DIR_ARG = 256, LIB_DIR_ARG, ENABLE_STD_ARG, DISABLE_STD_ARG
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

  opterr = 0;  // handle error reporting ourselves
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

  /* remaining argument is input file */
  if (optind == argc - 1) {
    input_file = find(include_dirs, argv[optind]).string();
  } else if (optind == argc) {
    throw CompilerException("No input file.");
  } else {
    throw CompilerException("Multiple input files.");
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
  if (!input_file.empty()) {
    queue(input_file.string(), enable_std);
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
  Importer importer;
  bool haveNew;

  /* first pass: populate available types */
  for (auto file : files) {
    Typer pass1;
    file->accept(&pass1);
  }

  /* propagate available types through graph of file imports */
  do {
    haveNew = false;
    for (auto file : files) {
      haveNew = importer.import(file) || haveNew;
    }
  } while (haveNew);

  /* second pass: populate available functions */
  for (auto file : files) {
    ResolverHeader pass2;
    file->accept(&pass2);
  }

  /* propagate available functions through graph of file imports */
  do {
    haveNew = false;
    for (auto file : files) {
      haveNew = importer.import(file) || haveNew;
    }
  } while (haveNew);

  /* third pass: resolve the bodies of functions in the input file */
  if (!input_file.empty()) {
    ResolverSource pass3;
    filesByName[input_file.string()]->accept(&pass3);
  }
}

void bi::Compiler::gen() {
  fs::path biPath, hppPath, cppPath;
  biPath = input_file;
  if (!output_file.empty()) {
    cppPath = output_file;
    hppPath = output_file;
  } else {
    cppPath = input_file;
    hppPath = input_file;
  }
  hppPath.replace_extension(".hpp");
  cppPath.replace_extension(".cpp");

  fs::ifstream biStream(biPath);
  fs::ofstream hppStream(hppPath);
  fs::ofstream cppStream(cppPath);

  hpp_ostream hppOutput(hppStream);
  cpp_ostream cppOutput(cppStream);

  hppOutput << filesByName[input_file.string()];

  /* the original file is also appended to the header, this ensures that any
   * changes to the *.bi file produce a different *.hpp file, even if those
   * changes would not otherwise produce different C++ code; `install -p` can
   * then be used when installing *.bi and *.hpp header files, and the last
   * modified date of the *.hpp header will never be earlier than the *.bi
   * header, important so that `make` does try to rebuild the *.hpp file */
  hppOutput << "\n// Original file\n";
  hppOutput << "#if 0\n";
  hppOutput.append(biStream);
  hppOutput << "\n#endif\n";

  cppOutput << filesByName[input_file.string()];
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
  return queue(path->file(), std);
}

bi::File* bi::Compiler::queue(const std::string name, const bool std) {
  std::string canonical = find(include_dirs, name).string();
  if (filesByName.find(canonical) == filesByName.end()) {
    files.push_back(new File(canonical));
    filesByName.insert(std::make_pair(canonical, files.back()));
    stds.insert(std::make_pair(canonical, std));
    unparsed.insert(canonical);
  }
  return filesByName[canonical];
}

void bi::Compiler::parse(const std::string name) {
  /* pre-condition */
  assert(unparsed.find(name) != unparsed.end());

  compiler = this;  // set global variable needed by parser for callbacks
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
  compiler = nullptr;
}
