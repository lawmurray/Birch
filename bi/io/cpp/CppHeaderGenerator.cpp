/**
 * @file
 */
#include "bi/io/cpp/CppHeaderGenerator.hpp"

#include "bi/io/cpp/CppForwardGenerator.hpp"
#include "bi/io/cpp/CppAliasGenerator.hpp"
#include "bi/io/cpp/CppSuperGenerator.hpp"
#include "bi/io/cpp/CppBaseGenerator.hpp"
#include "bi/visitor/Gatherer.hpp"
#include "bi/primitive/poset.hpp"
#include "bi/primitive/definitely.hpp"

#include "boost/filesystem.hpp"

bi::CppHeaderGenerator::CppHeaderGenerator(std::ostream& base,
    const int level) :
    indentable_ostream(base, level) {
  //
}

void bi::CppHeaderGenerator::visit(const Package* o) {
  CppForwardGenerator auxForward(base, level);
  CppAliasGenerator auxAlias(base, level);
  CppSuperGenerator auxSuper(base, level);
  CppBaseGenerator aux(base, level, true);

  line("#include \"bi/libbirch.hpp\"");
  for (auto header : o->headers) {
    boost::filesystem::path include = header->path;
    include.replace_extension(".hpp");
    line("#include \"" << include.string() << "\"");
  }

  /* raw C++ code for headers */
  Gatherer<Raw> raws;
  for (auto source: o->sources) {
    source->accept(&raws);
  }
  for (auto o1 : raws) {
    aux << o1;
  }
  line("");
  line("namespace bi {");

  /* forward class declarations */
  for (auto file : o->sources) {
    auxForward << file;
  }
  line("");

  /* typedef declarations */
  for (auto file : o->sources) {
    auxAlias << file;
  }
  line("");

  /* forward super type declarations */
  for (auto file : o->sources) {
    auxSuper << file;
  }
  line("");

  /* class definnitions; even with the forward declarations above, base
   * classes must be defined before their derived classes, so these are
   * gathered and sorted first */
  Gatherer<Class> classes;
  for (auto source: o->sources) {
    source->accept(&classes);
  }
  poset<Type*,definitely> sorted;
  for (auto type : classes) {
    sorted.insert(new ClassType(type));
  }
  for (auto type = sorted.rbegin(); type != sorted.rend(); ++type) {
    aux << dynamic_cast<ClassType*>(*type)->target;
  }

  /* global variable declarations */
  Gatherer<GlobalVariable> globals;
  for (auto source: o->sources) {
    source->accept(&globals);
  }
  for (auto o1 : globals) {
    aux << o1;
  }
  line("");

  /* function and fiber declarations */
  line("namespace func {");
  Gatherer<Function> functions;
  for (auto source: o->sources) {
    source->accept(&functions);
  }
  for (auto o1 : functions) {
    aux << o1;
  }
  Gatherer<Fiber> fibers;
  for (auto source: o->sources) {
    source->accept(&fibers);
  }
  for (auto o1 : fibers) {
    aux << o1;
  }
  line("}\n");

  /* programs */
  Gatherer<Program> programs;
  for (auto source: o->sources) {
    source->accept(&programs);
  }
  for (auto o1 : programs) {
    aux << o1;
  }
  line("");
  line("}\n");

  /* operators */
  Gatherer<BinaryOperator> binaries;
  for (auto source: o->sources) {
    source->accept(&binaries);
  }
  for (auto o1 : binaries) {
    aux << o1;
  }
  Gatherer<UnaryOperator> unaries;
  for (auto source: o->sources) {
    source->accept(&unaries);
  }
  for (auto o1 : unaries) {
    aux << o1;
  }
}
