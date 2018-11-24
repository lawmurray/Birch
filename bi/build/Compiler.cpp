/**
 * @file
 */
#include "Compiler.hpp"

#include "bi/visitor/Resolver.hpp"
#include "bi/io/cpp/CppPackageGenerator.hpp"
#include "bi/io/bi_ostream.hpp"
#include "bi/io/cpp_ostream.hpp"
#include "bi/io/hpp_ostream.hpp"
#include "bi/lexer.hpp"

#include <getopt.h>
#include <dlfcn.h>

bi::Compiler* compiler = nullptr;
std::stringstream raw;

bi::Compiler::Compiler(Package* package, const fs::path& build_dir,
    const bool unity) :
    scope(new Scope(GLOBAL_SCOPE)),
    package(package),
    build_dir(build_dir),
    unity(unity) {
  //
}

void bi::Compiler::parse() {
  compiler = this;  // set global variable needed by parser for callbacks
  for (auto file : package->files) {
    yyin = fopen(file->path.c_str(), "r");
    if (!yyin) {
      throw FileNotFoundException(file->path);
    }
    this->file = file;  // member variable needed by GNU Bison parser
    yyreset();
    do {
      try {
        yyparse();
      } catch (bi::Exception& e) {
        yyerror(e.msg.c_str());
      }
    } while (!feof(yyin));
    fclose(yyin);
    this->file = nullptr;
  }
  compiler = nullptr;
}

void bi::Compiler::resolve() {
  Resolver resolver;
  package->accept(&resolver);
}

void bi::Compiler::gen() {
  fs::path path;
  std::stringstream stream;

  bih_ostream bihOutput(stream);
  cpp_ostream cppOutput(stream);

  CppPackageGenerator hppPackageOutput(stream, 0, true);
  CppPackageGenerator cppPackageOutput(stream, 0, false);

  /* single *.bih header for whole package */
  stream.str("");
  bihOutput << package;
  path = build_dir / "bi" / tarname(package->name);
  path.replace_extension(".bih");
  write_all_if_different(path, stream.str());

  /* single *.hpp header for whole package */
  stream.str("");
  hppPackageOutput << package;
  path = build_dir / "bi" / tarname(package->name);
  path.replace_extension(".hpp");
  write_all_if_different(path, stream.str());

  /* *.cpp source for generic class specialization definitions */
  stream.str("");
  cppPackageOutput << package;
  if (unity) {
    /* for a unity build, C++ for all files goes into the one package
     * file */
    for (auto file : package->sources) {
      cppOutput << file;
    }
  }
  path = build_dir / "bi" / tarname(package->name);
  path.replace_extension(".cpp");
  write_all_if_different(path, stream.str());

  /* if unity mode is enabled, then C++ for each each *.bi file is
   * generated into the same package *.cpp file, otherwise one *.cpp file for
   * each *.bi file */
  if (!unity) {
    for (auto file : package->sources) {
      stream.str("");
      cppOutput << file;
      path = build_dir / file->path;
      path.replace_extension(".cpp");
      write_all_if_different(path, stream.str());
    }
  }
}

void bi::Compiler::setRoot(Statement* root) {
  this->file->root = root;
}
