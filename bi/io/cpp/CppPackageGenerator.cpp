/**
 * @file
 */
#include "bi/io/cpp/CppPackageGenerator.hpp"

#include "bi/io/cpp/CppRawGenerator.hpp"
#include "bi/io/cpp/CppClassGenerator.hpp"
#include "bi/visitor/Gatherer.hpp"
#include "bi/primitive/poset.hpp"
#include "bi/primitive/inherits.hpp"
#include "bi/build/misc.hpp"

bi::CppPackageGenerator::CppPackageGenerator(std::ostream& base,
    const int level, const bool header) :
    CppBaseGenerator(base, level, header) {
  //
}

void bi::CppPackageGenerator::visit(const Package* o) {
  /* gather important objects */
  Gatherer<Basic> basics;
  Gatherer<Class> classes, headerClasses;
  Gatherer<GlobalVariable> globals;
  Gatherer<Function> functions, headerFunctions;
  Gatherer<Fiber> fibers, headerFibers;
  Gatherer<Program> programs;
  Gatherer<BinaryOperator> binaries;
  Gatherer<UnaryOperator> unaries;
  for (auto file : o->sources) {
    file->accept(&basics);
    file->accept(&classes);
    file->accept(&globals);
    file->accept(&functions);
    file->accept(&fibers);
    file->accept(&programs);
    file->accept(&binaries);
    file->accept(&unaries);
  }
  for (auto file : o->headers) {
    file->accept(&headerClasses);
    file->accept(&headerFunctions);
    file->accept(&headerFibers);
  }

  /* base classes must be defined before their derived classes, so these are
   * gathered and sorted first */
  poset<const Class*,inherits> sortedClasses;
  for (auto o : classes) {
    if (!o->isGeneric()) {
      sortedClasses.insert(o);
    }
  }

  if (header) {
    /* a `#define` include guard is preferred to `#pragma once`; the header of
     * each package is included in sources with the `-include` compile option
     * rather than `#include` preprocessor directive, including (for
     * convenience/laziness) when precompiling the header itself; this seems
     * to cause a double inclusion with `#pragma once` but not with a `#define`
     * include guard */
    line("#ifndef BI_" << tarname(o->name) << "_HPP_");
    line("#define BI_" << tarname(o->name) << "_HPP_");
    line("");
    line("#include \"libbirch/libbirch.hpp\"");
    line("");

    for (auto header : o->headers) {
      fs::path include = header->path;
      include.replace_extension(".hpp");
      line("#include \"" << include.string() << "\"");
    }

    /* raw C++ code for headers */
    CppRawGenerator auxRaw(base, level, header);
    auxRaw << o;

    line("");
    line("namespace bi {");
    line("namespace type {");

    /* forward class declarations */
    for (auto o : classes) {
      if (!o->isAlias()) {
        genTemplateParams(o);
        line("class " << o->name << ';');
      }
    }
    line("");

    /* basic type aliases */
    for (auto o : basics) {
      if (o->isAlias()) {
        genTemplateParams(o);
        line("using " << o->name << " = " << o->base << ';');
      }
    }
    line("");

    /* class type aliases */
    for (auto o : classes) {
      if (o->isAlias()) {
        genTemplateParams(o);
        start("using " << o->name << " = " << o->base << ';');
      }
    }

    /* class definitions */
    for (auto o : sortedClasses) {
      assert(!o->isGeneric());
      if (!o->isAlias()) {
        *this << o;
      }
    }
    for (auto o : sortedClasses) {
      assert(!o->isGeneric());
      if (o->isAlias()) {
        *this << o;
      }
    }
    for (auto o : classes) {
      if (o->isGeneric() && !o->isAlias()) {
        *this << o;
      }
    }
    for (auto o : classes) {
      if (o->isGeneric() && o->isAlias()) {
        *this << o;
      }
    }
    line("");
    line("}\n");

    /* global variables */
    for (auto o : globals) {
      *this << o;
    }

    /* functions and fibers */
    for (auto o : functions) {
      *this << o;
    }
    for (auto o : fibers) {
      *this << o;
    }

    /* programs */
    for (auto o : programs) {
      *this << o;
    }

    /* operators */
    for (auto o : binaries) {
      *this << o;
    }
    for (auto o : unaries) {
      *this << o;
    }

    line("}\n");
    line("");
    line("#endif");
  }
}
