/**
 * @file
 */
#include "src/generate/CppPackageGenerator.hpp"

#include "src/generate/CppClassGenerator.hpp"
#include "src/visitor/Gatherer.hpp"
#include "src/primitive/poset.hpp"
#include "src/primitive/inherits.hpp"
#include "src/primitive/string.hpp"
#include "src/primitive/system.hpp"

birch::CppPackageGenerator::CppPackageGenerator(std::ostream& base,
    const int level, const bool header) :
    CppGenerator(base, level, header, false) {
  //
}

void birch::CppPackageGenerator::visit(const Package* o) {
  /* auxiliary generators */
  CppGenerator auxDeclaration(base, level, true, true);
  CppGenerator auxDefinition(base, level, false, true);

  /* gather important objects */
  Gatherer<Basic> basics;
  Gatherer<Class> classes;
  Gatherer<GlobalVariable> globals;
  Gatherer<Function> functions;
  Gatherer<Program> programs;
  Gatherer<BinaryOperator> binaries;
  Gatherer<UnaryOperator> unaries;
  for (auto file : o->sources) {
    file->accept(&basics);
    file->accept(&classes);
    file->accept(&globals);
    file->accept(&functions);
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
    /* don't use #pragma once here, use a macro guard instead, as the header
     * may be used as a source file to create a pre-compiled header */
    std::string name = upper(canonical(o->name));
    line("#ifndef BI_" << name << "_HPP");
    line("#define BI_" << name << "_HPP\n");
    line("#include \"libbirch.hpp\"\n");

    for (auto name : o->packages) {
      fs::path include(tar(name));
      include.replace_extension(".hpp");
      line("#include \"" << include.string() << "\"");
    }

    /* raw C++ code for headers */
    for (auto file : o->sources) {
      for (auto o : *file->root) {
        auto raw = dynamic_cast<const Raw*>(o);
        if (raw) {
          *this << raw;
        }
      }
    }

    line("");
    line("namespace birch {");
    line("namespace type {");

    /* forward class type declarations */
    for (auto o : classes) {
      if (!o->isAlias()) {
        genTemplateParams(o);
        if (o->has(STRUCT)) {
          start("struct ");
        } else {
          start("class ");
        }
        finish(o->name << ';');
      }
    }
    line("");

    /* basic type aliases */
    for (auto o : basics) {
      if (o->isAlias()) {
        auto base = dynamic_cast<const NamedType*>(o->base);
        assert(base);
        genTemplateParams(o);
        genSourceLine(o->loc);
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
        genSourceLine(o->loc);
        start("using " << o->name << " = " << base->name);
        if (!base->typeArgs->isEmpty()) {
          middle('<' << base->typeArgs << '>');
        }
        finish(';');
      }
    }

    /* classes */
    for (auto o : sortedClasses) {
      if (!o->isAlias()) {
        auxDeclaration << o;
      }
    }

    line("");
    line("}\n");

    /* global variables */
    for (auto o : globals) {
      auxDeclaration << o;
    }

    /* functions */
    for (auto o : functions) {
      auxDeclaration << o;
    }

    /* binary operators */
    for (auto o : binaries) {
      auxDeclaration << o;
    }

    /* unary operators */
    for (auto o : unaries) {
      auxDeclaration << o;
    }

    /* programs */
    for (auto o : programs) {
      auxDeclaration << o;
    }

    line("}\n");

    /* generic class type definitions, generic member definitions */
    for (auto o : classes) {
      if (o->isGeneric() && !o->isAlias()) {
        /* whole class (which may include generic members) */
        auxDefinition << o;
      } else {
        /* just generic members of the class */
        CppClassGenerator auxMember(base, level, false, true, o);

        Gatherer<MemberFunction> memberFunctions;
         o->accept(&memberFunctions);
        for (auto o : memberFunctions) {
          if (o->isGeneric()) {
            auxMember << o;
          }
        }
      }
    }

    /* generic function and operator definitions */
    for (auto o : functions) {
      if (o->isGeneric()) {
        auxDefinition << o;
      }
    }
    for (auto o : binaries) {
      if (o->isGeneric()) {
        auxDefinition << o;
      }
    }
    for (auto o : unaries) {
      if (o->isGeneric()) {
        auxDefinition << o;
      }
    }

    line("#endif");
  }
}
