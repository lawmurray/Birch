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
    const int level, const bool header, const bool includeInlines,
    const bool includeLines) :
    CppGenerator(base, level, header, includeInlines, includeLines) {
  //
}

void birch::CppPackageGenerator::visit(const Package* o) {
  /* auxiliary generators */
  CppGenerator auxDeclaration(base, level, true, true, includeLines);
  CppGenerator auxDefinition(base, level, false, true, includeLines);

  /* gather important objects */
  Gatherer<Basic> basics;
  Gatherer<Struct> structs;
  Gatherer<Class> classes;
  Gatherer<GlobalVariable> globals;
  Gatherer<Function> functions;
  Gatherer<Program> programs;
  Gatherer<BinaryOperator> binaries;
  Gatherer<UnaryOperator> unaries;
  for (auto file : o->sources) {
    file->accept(&basics);
    file->accept(&structs);
    file->accept(&classes);
    file->accept(&globals);
    file->accept(&functions);
    file->accept(&programs);
    file->accept(&binaries);
    file->accept(&unaries);
  }

  /* base classes must be defined before their derived classes, so these are
   * gathered and sorted first */
  poset<const Struct*,inherits> sortedStructs;
  for (auto o : structs) {
    sortedStructs.insert(o);
  }
  poset<const Class*,inherits> sortedClasses;
  for (auto o : classes) {
    sortedClasses.insert(o);
  }

  if (header) {
    /* don't use #pragma once here, use a macro guard instead, as the header
     * may be used as a source file to create a pre-compiled header */
    std::string name = upper(canonical(o->name));
    line("#ifndef " << name << "_HPP");
    line("#define " << name << "_HPP\n");
    line("#include <numbirch.hpp>");
    line("#include <membirch.hpp>\n");

    for (auto name : o->packages) {
      fs::path include(tar(name));
      include.replace_extension(".hpp");
      line("#include <" << include.string() << '>');
    }

    /* raw C++ code */
    for (auto file : o->sources) {
      for (auto o : *file->root) {
        auto raw = dynamic_cast<const Raw*>(o);
        if (raw) {
          *this << raw;
        }
      }
    }

    line("namespace birch {");

    /* forward struct type declarations */
    for (auto o : structs) {
      if (!o->isAlias()) {
        genTemplateParams(o);
        genSourceLine(o->loc);
        line("struct " << o->name << ';');
      }
    }
    line("");

    /* forward class type declarations */
    for (auto o : classes) {
      if (!o->isAlias()) {
        genTemplateParams(o);
        genSourceLine(o->loc);
        line("class " << o->name << "_;");
        genTemplateParams(o);
        genSourceLine(o->loc);
        start("using " << o->name << " = membirch::Shared<" << o->name << '_');
        genTemplateArgs(o);
        finish(">;");
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

    /* struct type aliases */
    for (auto o : structs) {
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

    /* global variables */
    for (auto o : globals) {
      auxDeclaration << o;
    }

    /* structs */
    for (auto o : sortedStructs) {
      if (!o->isAlias()) {
        auxDeclaration << o;
      }
    }

    /* classes */
    for (auto o : sortedClasses) {
      if (!o->isAlias()) {
        auxDeclaration << o;
      }
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

    /* generic struct definitions */
    for (auto o : structs) {
      if (o->isGeneric() && !o->isAlias()) {
        auxDefinition << o;
      }
    }

    /* generic class definitions, generic member definitions */
    for (auto o : classes) {
      if (o->isGeneric() && !o->isAlias()) {
        /* whole class (which may include generic members) */
        auxDefinition << o;
      } else {
        /* just generic members of the class */
        CppClassGenerator auxMember(base, level, false, true, includeLines,
            o);

        Gatherer<MemberFunction> memberFunctions;
         o->accept(&memberFunctions);
        for (auto o : memberFunctions) {
          if (o->isGeneric()) {
            auxMember << o;
          }
        }
      }
    }

    /* generic function and operator definitions; those with deduced return
     * types first to reduce occurrences of use before deduction */
    for (auto o : functions) {
      if (o->isGeneric() && o->returnType->isDeduced()) {
        auxDefinition << o;
      }
    }
    for (auto o : binaries) {
      if (o->isGeneric() && o->returnType->isDeduced()) {
        auxDefinition << o;
      }
    }
    for (auto o : unaries) {
      if (o->isGeneric() && o->returnType->isDeduced()) {
        auxDefinition << o;
      }
    }
    for (auto o : functions) {
      if (o->isGeneric() && !o->returnType->isDeduced()) {
        auxDefinition << o;
      }
    }
    for (auto o : binaries) {
      if (o->isGeneric() && !o->returnType->isDeduced()) {
        auxDefinition << o;
      }
    }
    for (auto o : unaries) {
      if (o->isGeneric() && !o->returnType->isDeduced()) {
        auxDefinition << o;
      }
    }

    line("}\n");  // close namespace
    line("#endif");
  }
}
