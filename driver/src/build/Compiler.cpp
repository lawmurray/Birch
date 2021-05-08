/**
 * @file
 */
#include "src/build/Compiler.hpp"

#include "src/birch.hpp"
#include "src/lexer.hpp"
#include "src/visitor/all.hpp"
#include "src/generate/CppGenerator.hpp"
#include "src/generate/CppPackageGenerator.hpp"
#include "src/primitive/string.hpp"

birch::Compiler* compiler = nullptr;
std::stringstream raw;

birch::Compiler::Compiler(Package* package, const std::string& unit) :
    scope(new Scope(GLOBAL_SCOPE)),
    package(package),
    unit(unit) {
  //
}

void birch::Compiler::parse() {
  compiler = this;  // set global variable needed by parser for callbacks
  auto files = package->sources;
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
    } catch (birch::Exception& e) {
      yyerror(e.msg.c_str());
    }
    yylex_destroy();
    fclose(fd);
    this->file = nullptr;
  }
  compiler = nullptr;
}

void birch::Compiler::gen() {
  std::stringstream stream;
  std::string tarName = tar(package->name);
  fs::path path = fs::path(tarName);

  CppPackageGenerator hppOutput(stream, 0, true);
  CppGenerator cppOutput(stream, 0, false, false);

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
    /* sources go into one *.cpp file for each *.birch file */
    for (auto file : package->sources) {
      stream.str("");
      cppOutput << file;
      path = file->path;
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
      path = fs::path(pair.first) / tarName;
      path.replace_extension(".cpp");
      write_all_if_different(path, pair.second);
    }
  }
}

void birch::Compiler::setRoot(Statement* root) {
  this->file->root = root;
}
