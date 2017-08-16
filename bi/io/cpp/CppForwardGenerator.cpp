/**
 * @file
 */
#include "bi/io/cpp/CppForwardGenerator.hpp"

#include "bi/io/cpp/CppBaseGenerator.hpp"

bi::CppForwardGenerator::CppForwardGenerator(std::ostream& base,
    const int level) :
    indentable_ostream(base, level) {
  //
}

void bi::CppForwardGenerator::visit(const File* o) {
  line("namespace bi {");
  in();
  line("namespace type {");
  out();
  Visitor::visit(o);
  in();
  line("}");
  out();
  line("}\n");
}

void bi::CppForwardGenerator::visit(const Class* o) {
  Visitor::visit(o);
  if (classes.find(o) == classes.end()) {
    classes.insert(o);
    start("class ");
    CppBaseGenerator aux(base, level);
    aux << o->name;
    finish(';');
  }
}

void bi::CppForwardGenerator::visit(const Alias* o) {
  Visitor::visit(o);
  if (aliases.find(o) == aliases.end()) {
    aliases.insert(o);
    start("using ");
    CppBaseGenerator aux(base, level);
    aux << o->name << " = " << o->base;
    finish(';');
  }
}
