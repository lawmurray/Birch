/**
 * @file
 */
#include "bi/io/cpp/CppHeaderGenerator.hpp"

#include "bi/io/cpp/CppForwardGenerator.hpp"
#include "bi/io/cpp/CppAliasGenerator.hpp"
#include "bi/io/cpp/CppSuperGenerator.hpp"
#include "bi/io/cpp/CppBaseGenerator.hpp"
#include "bi/visitor/Gatherer.hpp"

bi::CppHeaderGenerator::CppHeaderGenerator(std::ostream& base,
    const int level) :
    indentable_ostream(base, level) {
  //
}

void bi::CppHeaderGenerator::visit(const Package* o) {
  line("#include \"bi/libbirch.hpp\"");
  line("#ifdef ENABLE_STD");
  line("#include \"bi/birch_standard.hpp\"");
  line("#endif\n");
  line("namespace bi {");

  CppForwardGenerator auxForward(base, level);
  for (auto file : o->sources) {
    auxForward << file;
  }
  line("");

  CppAliasGenerator auxAlias(base, level);
  for (auto file : o->sources) {
    auxAlias << file;
  }
  line("");

  CppSuperGenerator auxSuper(base, level);
  for (auto file : o->sources) {
    auxSuper << file;
  }
  line("");

  CppBaseGenerator aux(base, level, true);

  Gatherer<Class> classes;
  o->accept(&classes);
  for (auto o1 : classes) {
    aux << o1;
  }

  Gatherer<GlobalVariable> globals;
  o->accept(&globals);
  for (auto o1 : globals) {
    aux << o1;
  }
  line("");

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
