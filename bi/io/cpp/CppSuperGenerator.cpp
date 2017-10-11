/**
 * @file
 */
#include "bi/io/cpp/CppSuperGenerator.hpp"

#include "bi/io/cpp/CppBaseGenerator.hpp"

bi::CppSuperGenerator::CppSuperGenerator(std::ostream& base,
    const int level) :
    indentable_ostream(base, level) {
  //
}

void bi::CppSuperGenerator::visit(const Class* o) {
  if (!o->braces->isEmpty()) {
    CppBaseGenerator aux(base, level);
    start("template<> struct super_type<");
    aux << o->name;
    finish("> {");
    in();
    start("typedef ");
    if (!o->base->isEmpty()) {
      auto super = dynamic_cast<const ClassType*>(o->base);
      assert(super);
      aux << super->name;
    } else {
      aux << "Object_";
    }
    finish(" type;");
    out();
    line("};");
  }
}
