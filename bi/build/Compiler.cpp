/**
 * @file
 */
#include "Compiler.hpp"

#include "bi/build/misc.hpp"
#include "bi/visitor/Typer.hpp"
#include "bi/visitor/ResolverSuper.hpp"
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

bi::Compiler::Compiler(Package* package,
    const boost::filesystem::path& work_dir,
    const boost::filesystem::path& build_dir) :
    package(package),
    work_dir(work_dir),
    build_dir(build_dir),
    scope(new Scope()) {
  //
}

void bi::Compiler::parse() {
  compiler = this;  // set global variable needed by parser for callbacks
  for (auto file : package->files) {
    std::string name = (work_dir / file->path).string();
    yyin = fopen(name.c_str(), "r");
    if (!yyin) {
      throw FileNotFoundException(name);
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
  /* first pass: populate available types */
  for (auto file : package->files) {
    Typer pass1;
    file->accept(&pass1);
  }

  /* second pass: resolve super type relationships */
  for (auto file : package->files) {
    ResolverSuper pass2;
    file->accept(&pass2);
  }

  /* third pass: populate available functions */
  for (auto file : package->files) {
    ResolverHeader pass3;
    file->accept(&pass3);
  }

  /* fourth pass: resolve the bodies of functions */
  for (auto file : package->sources) {
    ResolverSource pass4;
    file->accept(&pass4);
  }
}

void bi::Compiler::gen() {
  fs::path bihPath, biPath, hppPath, cppPath;

  /* single *.bih header for whole package */
  bihPath = build_dir / "bi" / package->tarname;
  bihPath.replace_extension(".bih");
  boost::filesystem::create_directories(bihPath.parent_path());
  fs::ofstream bihStream(bihPath);
  bih_ostream bihOutput(bihStream);
  bihOutput << package;

  /* single *.hpp header for whole package */
  hppPath = build_dir / "bi" / package->tarname;
  hppPath.replace_extension(".hpp");
  boost::filesystem::create_directories(hppPath.parent_path());
  fs::ofstream hppStream(hppPath);
  hpp_ostream hppOutput(hppStream);
  hppOutput << package;

  /* separate *.cpp source for each file */
  for (auto file : package->sources) {
    biPath = work_dir / file->path;
    cppPath = build_dir / file->path;
    cppPath.replace_extension(".cpp");
    boost::filesystem::create_directories(cppPath.parent_path());
    fs::ofstream cppStream(cppPath);
    cpp_ostream cppOutput(cppStream);
    cppOutput << file;
  }
}

void bi::Compiler::setRoot(Statement* root) {
  this->file->root = root;
}
