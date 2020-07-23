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
  CppBaseGenerator auxDeclaration(base, level, true, true);
  CppBaseGenerator auxDefinition(base, level, false, true);

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
    /* don't use #pragma once here, use a macro guard instead, as the header
     * may be used as a source file to create a pre-compiled header */
    std::string name = tarname(o->name);
    boost::to_upper(name);
    line("#ifndef BI_" << name << "_HPP");
    line("#define BI_" << name << "_HPP\n");
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

    /* fibers */
    for (auto o : fibers) {
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

        Gatherer<MemberFiber> memberFibers;
        o->accept(&memberFibers);
        for (auto o : memberFibers) {
          if (o->isGeneric()) {
            auxMember << o;
          }
        }
      }
    }

    /* generic function definitions */
    for (auto o : functions) {
      if (o->isGeneric()) {
        auxDefinition << o;
      }
    }

    /* generic fiber definitions */
    for (auto o : fibers) {
      if (o->isGeneric()) {
        auxDefinition << o;
      }
    }
  }

  if (header) {
    line("#endif");
  }
}
