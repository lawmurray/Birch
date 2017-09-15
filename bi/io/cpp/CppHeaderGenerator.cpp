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
  line("#ifdef ENABLE_STD");
  line("#include \"bi/birch_standard.hpp\"");
  line("#endif\n");
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
  o->accept(&classes);
  poset<Type*,definitely> sorted;
  for (auto type : classes) {
    sorted.insert(new ClassType(type));
  }
  for (auto type = sorted.rbegin(); type != sorted.rend(); ++type) {
    aux << dynamic_cast<ClassType*>(*type)->target;
  }

  /* global variable declarations */
  Gatherer<GlobalVariable> globals;
  o->accept(&globals);
  for (auto o1 : globals) {
    aux << o1;
  }
  line("");

  /* function declarations */
  line("namespace func {");
  Gatherer<Function> functions;
  o->accept(&functions);
  for (auto o1 : functions) {
    aux << o1;
  }
  Gatherer<Fiber> fibers;
  o->accept(&fibers);
  for (auto o1 : fibers) {
    aux << o1;
  }
  line("}\n");

  line("}\n");
}
