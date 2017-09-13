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
  Visitor::visit(o);
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
    if (!o->base->isEmpty()) {
      auto type = dynamic_cast<const ClassType*>(o->base);
      assert(type);

      /* forward super type declaration */
      start("template<> struct super_type<");
      aux << o->name;
      finish("> {");
      in();
      start("typedef class ");
      aux << type->name;
      finish(" type;");
      out();
      line("};");
    }
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
