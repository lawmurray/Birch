/**
 * @file
 */
#include "bi/io/cpp/CppHeaderGenerator.hpp"

#include "bi/io/cpp/CppRawGenerator.hpp"
#include "bi/visitor/Gatherer.hpp"
#include "bi/primitive/poset.hpp"
#include "bi/primitive/definitely.hpp"
#include "bi/build/misc.hpp"

#include "boost/filesystem.hpp"

bi::CppHeaderGenerator::CppHeaderGenerator(std::ostream& base,
    const int level, const bool header) :
    CppBaseGenerator(base, level, header) {
  //
}

void bi::CppHeaderGenerator::visit(const Package* o) {
  /* a `#define` include guard is preferred to `#pragma once`; the header of
   * each package is included in sources with the `-include` compile option
   * rather than `#include` preprocessor directive, including (for
   * convenience/laziness) when precompiling the header itself; this seems
   * to cause a double inclusion with `#pragma once` but not with a `#define`
   * include guard */
  line("#ifndef BI_" << tarname(o->name) << "_HPP");
  line("#define BI_" << tarname(o->name) << "_HPP");
  line("");

  line("#include \"libbirch/libbirch.hpp\"");
  for (auto header : o->headers) {
    boost::filesystem::path include = header->path;
    include.replace_extension(".hpp");
    line("#include \"" << include.string() << "\"");
  }

  /* gather important objects */
  Gatherer<Basic> basics;
  Gatherer<Class> classes;
  Gatherer<GlobalVariable> globals;
  Gatherer<Function> functions;
  Gatherer<Fiber> fibers;
  Gatherer<Program> programs;
  Gatherer<BinaryOperator> binaries;
  Gatherer<UnaryOperator> unaries;
  for (auto source : o->sources) {
    source->accept(&basics);
    source->accept(&classes);
    source->accept(&globals);
    source->accept(&functions);
    source->accept(&fibers);
    source->accept(&programs);
    source->accept(&binaries);
    source->accept(&unaries);
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

  /* type declarations */
  for (auto o : basics) {
    line("using " << o->name << " = " << o->base << ';');
  }
  line("}\n");
  line("");

  /* forward super type declarations */
  for (auto o : classes) {
    if (!o->base->isEmpty()) {
      if (o->isGeneric()) {
        genTemplateParams(o);
      } else {
        start("template<>");
      }
      middle(" struct super_type<type::" << o->name);
      genTemplateArgs(o);
      finish("> {");
      in();
      line("using type = " << o->base << ';');
      out();
      line("};");
    }
  }
  line("}\n");
  line("");

  /* class definitions; even with the forward declarations above, base
   * classes must be defined before their derived classes, so these are
   * gathered and sorted first */
  poset<Type*,definitely> sorted;
  for (auto o : classes) {
    sorted.insert(new ClassType(o));
  }
  for (auto iter = sorted.rbegin(); iter != sorted.rend(); ++iter) {
    *this << (*iter)->getClass();
  }

  line("namespace bi {");

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
