/**
 * @file
 */
#include "Compiler.hpp"

#include "bi/birch.hpp"
#include "bi/lexer.hpp"
#include "bi/visitor/all.hpp"
#include "bi/io/bih_ostream.hpp"
#include "bi/io/cpp/CppBaseGenerator.hpp"
#include "bi/io/cpp/CppPackageGenerator.hpp"

bi::Compiler* compiler = nullptr;
std::stringstream raw;

bi::Compiler::Compiler(Package* package, const std::string& unit) :
    scope(new Scope(GLOBAL_SCOPE)),
    package(package),
    unit(unit) {
  //
}

void bi::Compiler::parse(bool includeHeaders) {
  compiler = this;  // set global variable needed by parser for callbacks
  auto files = package->sources;
  if (includeHeaders) {
    files = package->files;
  }
  for (auto file : files) {
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
  std::stringstream stream;
  std::string tarName = tar(package->name);
  fs::path path = fs::path("src") / tarName;

  bih_ostream bihOutput(stream);
  CppPackageGenerator hppOutput(stream, 0, true);
  CppBaseGenerator cppOutput(stream, 0, false, false);

  /* single *.bih header for whole package */
  stream.str("");
  bihOutput << package;
  path.replace_extension(".bih");
  write_all_if_different(path, stream.str());

  /* single *.hpp header for whole package */
  stream.str("");
  hppOutput << package;
  path.replace_extension(".hpp");
  write_all_if_different(path, stream.str());

  if (unit == "unity") {
    /* sources go into one *.cpp file for the whole package */
    stream.str("");
    for (auto file : package->sources) {
      cppOutput << file;
    }
    path.replace_extension(".cpp");
    write_all_if_different(path, stream.str());
  } else if (unit == "file") {
    /* sources go into one *.cpp file for each *.bi file */
    for (auto file : package->sources) {
      stream.str("");
      cppOutput << file;
      path = fs::path("src") / file->path;
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
      path = fs::path("src") / pair.first / tarName;
      path.replace_extension(".cpp");
      write_all_if_different(path, pair.second);
    }
  }
}

void bi::Compiler::setRoot(Statement* root) {
  this->file->root = root;
}
