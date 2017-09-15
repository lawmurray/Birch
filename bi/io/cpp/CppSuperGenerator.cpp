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
  if (!o->base->isEmpty()) {
    auto super = dynamic_cast<const ClassType*>(o->base);
    assert(super);

    CppBaseGenerator aux(base, level);
    start("template<> struct super_type<");
    aux << o->name;
    finish("> {");
    in();
    start("typedef ");
    aux << super->name;
    finish(" type;");
    out();
    line("};");
  }
}
