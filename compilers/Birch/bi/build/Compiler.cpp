/**
 * @file
 */
#include "Compiler.hpp"

#include "bi/birch.hpp"
#include "bi/visitor/all.hpp"
#include "bi/io/cpp/CppPackageGenerator.hpp"
#include "bi/io/bi_ostream.hpp"
#include "bi/io/cpp_ostream.hpp"
#include "bi/io/hpp_ostream.hpp"
#include "bi/lexer.hpp"

bi::Compiler* compiler = nullptr;
std::stringstream raw;

bi::Compiler::Compiler(Package* package, const fs::path& build_dir,
    const std::string& unit) :
    scope(new Scope(GLOBAL_SCOPE)),
    package(package),
    build_dir(build_dir),
    unit(unit) {
  //
}

void bi::Compiler::parse() {
  compiler = this;  // set global variable needed by parser for callbacks
  for (auto file : package->files) {
    raw.str("");
    auto fd = fopen(file->path.c_str(), "r");
    if (!fd) {
      throw FileNotFoundException(file->path);
    }
    this->file = file;  // member variable needed by GNU Bison parser
    yyrestart(fd);
    yyreset();
    try {
      yyparse();
    } catch (bi::Exception& e) {
      yyerror(e.msg.c_str());
    }
    yylex_destroy();
    fclose(fd);
    this->file = nullptr;
  }
  compiler = nullptr;
}

void bi::Compiler::resolve() {
  Spinner spinner;
  package->accept(&spinner);

  Transformer transformer;
  package->accept(&transformer);

  Checker checker;
  package->accept(&checker);

  Scoper scoper;
  package->accept(&scoper);

  Baser baser;
  package->accept(&baser);

  Resolver resolver;
  package->accept(&resolver);
}

void bi::Compiler::gen() {
  fs::path path;
  std::stringstream stream;
  std::string internalName = tarname(package->name);

  bih_ostream bihOutput(stream);
  cpp_ostream cppOutput(stream, unit);

  CppPackageGenerator hppPackageOutput(stream, unit, 0, true);

  /* single *.bih header for whole package */
  stream.str("");
  bihOutput << package;
  path = build_dir / "bi" / internalName;
  path.replace_extension(".bih");
  write_all_if_different(path, stream.str());

  /* single *.hpp header for whole package */
  stream.str("");
  hppPackageOutput << package;
  path = build_dir / "bi" / internalName;
  path.replace_extension(".hpp");
  write_all_if_different(path, stream.str());

  if (unit == "unity") {
    /* sources go into one *.cpp file for the whole package */
    stream.str("");
    for (auto file : package->sources) {
      cppOutput << file;
    }
    path = build_dir / "bi" / internalName;
    path.replace_extension(".cpp");
    write_all_if_different(path, stream.str());
  } else if (unit == "file") {
    /* sources go into one *.cpp file for each *.bi file */
    for (auto file : package->sources) {
      stream.str("");
      cppOutput << file;
      path = build_dir / file->path;
      path.replace_extension(".cpp");
      write_all_if_different(path, stream.str());
    }
  } else {
    /* sources go into one *.cpp file for each directory */
    std::unordered_map<std::string,std::string> sources;
    for (auto file : package->sources) {
      auto dir = fs::path(file->path).parent_path().string();
      auto iter = sources.find(dir);
      if (iter == sources.end()) {
        iter = sources.insert(std::make_pair(dir, std::string(""))).first;
      }
      stream.str("");
      cppOutput << file;
      iter->second += stream.str();
    }
    for (auto pair : sources) {
      path = build_dir / pair.first / internalName;
      path.replace_extension(".cpp");
      write_all_if_different(path, pair.second);
    }
  }
}

void bi::Compiler::setRoot(Statement* root) {
  this->file->root = root;
}
