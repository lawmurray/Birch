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
    line("#pragma once\n");
    line("#include \"libbirch/libbirch.hpp\"\n");

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

    /* forward super type declarations */
    for (auto o : classes) {
      if (!o->isAlias()) {
        auto base = dynamic_cast<const NamedType*>(o->base);
        if (!o->typeParams->isEmpty()) {
          genTemplateParams(o);
        } else {
          line("template<>");
        }
        start("struct super_type<" << o->name);
        genTemplateArgs(o);
        finish("> {");
        in();
        start("using type = ");
        if (base) {
          middle(base->name);
          if (!base->typeArgs->isEmpty()) {
            middle('<' << base->typeArgs << '>');
          }
        } else {
          middle("libbirch::Any");
        }
        finish(';');
        out();
        line("};");
      }
    }

    /* class type declarations */
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
//  } else {
//    /* generic class type definitions */
//    for (auto o : classes) {
//      if (o->isGeneric() && !o->isAlias()) {
//        *this << o;
//      }
//    }
//
//    /* generic function definitions */
//    for (auto o : functions) {
//      if (o->isGeneric()) {
//        *this << o;
//      }
//    }
//
//    /* generic fiber definitions */
//    for (auto o : fibers) {
//      if (o->isGeneric()) {
//        *this << o;
//      }
//    }
  }
}
