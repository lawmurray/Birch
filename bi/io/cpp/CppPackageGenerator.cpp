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
  /* auxiliary generators */
  CppRawGenerator auxRaw(base, level, header);

  /* gather important objects */
  Gatherer<Basic> basics;
  Gatherer<Class> classes;
  Gatherer<GlobalVariable> globals;
  Gatherer<Function> functions;
  Gatherer<Fiber> fibers;
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

  /* base classes must be defined before their derived classes, so these are
   * gathered and sorted first */
  poset<const Class*,inherits> sortedClasses;
  for (auto o : classes) {
    sortedClasses.insert(o);
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
    auxRaw << o;

    line("");
    line("namespace bi {");
    line("namespace type {");

    /* forward class type declarations */
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
        auto base = dynamic_cast<const NamedType*>(o->base);
        assert(base);
        genTemplateParams(o);
        start("using " << o->name << " = " << base->name);
        if (!base->typeArgs->isEmpty()) {
          middle('<' << base->typeArgs << '>');
        }
        finish(';');
      }
    }
    line("");

    /* class type aliases */
    for (auto o : classes) {
      if (o->isAlias()) {
        auto base = dynamic_cast<const NamedType*>(o->base);
        assert(base);
        genTemplateParams(o);
        start("using " << o->name << " = " << base->name);
        if (!base->typeArgs->isEmpty()) {
          middle('<' << base->typeArgs << '>');
        }
        finish(';');
      }
    }

    /* non-generic class type declarations */
    for (auto o : sortedClasses) {
      if (!o->isAlias()) {
        *this << o;
      }
    }

    line("");
    line("}\n");

    /* global variables */
    for (auto o : globals) {
      *this << o;
    }

    /* functions */
    for (auto o : functions) {
      *this << o;
    }

    /* fibers */
    for (auto o : fibers) {
      *this << o;
    }

    /* binary operators */
    for (auto o : binaries) {
      *this << o;
    }

    /* unary operators */
    for (auto o : unaries) {
      *this << o;
    }

    /* programs */
    for (auto o : programs) {
      *this << o;
    }

    line("}\n");
  } else {
    /* generic class type definitions */
    for (auto o : classes) {
      if (o->isGeneric() && !o->isAlias()) {
        *this << o;
      }
    }

    /* generic function definitions */
    for (auto o : functions) {
      if (o->isGeneric()) {
        *this << o;
      }
    }

    /* generic fiber definitions */
    for (auto o : fibers) {
      if (o->isGeneric()) {
        *this << o;
      }
    }
  }

  line("");
  line("#endif");
}
