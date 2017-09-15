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

void bi::CppForwardGenerator::visit(const Class* o) {
  start("class ");
  CppBaseGenerator aux(base, level);
  aux << o->name;
  finish(';');
}
