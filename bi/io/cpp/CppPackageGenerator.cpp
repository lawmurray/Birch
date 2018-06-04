/**
 * @file
 */
#include "bi/io/cpp/CppPackageGenerator.hpp"

#include "bi/io/cpp/CppRawGenerator.hpp"
#include "bi/io/cpp/CppClassGenerator.hpp"
#include "bi/visitor/Gatherer.hpp"
#include "bi/primitive/poset.hpp"
#include "bi/primitive/definitely.hpp"
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
  for (auto file : o->headers) {
    file->accept(&headerClasses);
  }

  /* base classes must be defined before their derived classes, so these are
   * gathered and sorted first */
  poset<Type*,definitely> sorted;
  for (auto o : classes) {
    if (!o->isAlias()) {
      sorted.insert(new ClassType(o));
    }
    for (auto instantiation : o->instantiations) {
      if (!instantiation->isExplicit) {
        sorted.insert(new ClassType(instantiation));
      }
    }
  }
  for (auto o : headerClasses) {
    for (auto instantiation : o->instantiations) {
      if (!instantiation->isExplicit) {
        sorted.insert(new ClassType(instantiation));
      }
    }
  }
  std::list<Class*> sortedClasses;
  for (auto iter = sorted.rbegin(); iter != sorted.rend(); ++iter) {
    sortedClasses.push_back((*iter)->getClass());
  }

  if (header) {
    /* a `#define` include guard is preferred to `#pragma once`; the header of
     * each package is included in sources with the `-include` compile option
     * rather than `#include` preprocessor directive, including (for
     * convenience/laziness) when precompiling the header itself; this seems
     * to cause a double inclusion with `#pragma once` but not with a `#define`
     * include guard */
    line("#ifndef BI_" << tarname(o->name) << "_HPP");
    line("#define BI_" << tarname(o->name) << "_HPP");
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
      if (!o->braces->isEmpty()) {
        genTemplateParams(o);
        line("class " << o->name << ';');
      }
    }
    line("");

    /* basic type aliases */
    for (auto o : basics) {
      if (o->isAlias()) {
        line("using " << o->name << " = " << o->base << ';');
      }
    }
    line("");

    /* class type aliases */
    for (auto o : classes) {
      if (o->isAlias()) {
        line("using " << o->name << " = " << o->base << ';');
      }
    }

    line("}\n");
    line("");

    /* forward super type declarations */
    for (auto o : sortedClasses) {
      if (!o->base->isEmpty() && (!o->isGeneric() || o->isInstantiation)) {
        start("template<> ");
        middle("struct super_type<type::" << o->name);
        genTemplateArgs(o);
        finish("> {");
        in();
        line("using type = " << o->base << ';');
        out();
        line("};");
      }
    }

    /* forward assignment operator declarations */
    for (auto o : sortedClasses) {
      if (!o->isGeneric() || o->isInstantiation) {
        for (auto o1 : o->assignments) {
          start("template<> ");
          middle("struct has_assignment<type::" << o->name);
          genTemplateArgs(o);
          finish("," << o1 << "> {");
          in();
          line("static const bool value = true;");
          out();
          line("};");
        }
      }
    }

    /* forward conversion operator declarations */
    for (auto o : sortedClasses) {
      if (!o->isGeneric() || o->isInstantiation) {
        for (auto o1 : o->conversions) {
          start("template<> ");
          middle("struct has_conversion<type::" << o->name);
          genTemplateArgs(o);
          finish("," << o1 << "> {");
          in();
          line("static const bool value = true;");
          out();
          line("};");
        }
      }
    }

    /* class definitions */
    line("namespace type {");
    for (auto o : sortedClasses) {
      *this << o;
    }
    for (auto o : classes) {
      if (o->isAlias()) {
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
  } else {
    /* instantiations of generic classes go in the package source file */
    for (auto o : sortedClasses) {
      if (o->isInstantiation && !o->isExplicit) {
        *this << o;
      }
    }
  }
}
