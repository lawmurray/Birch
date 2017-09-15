/**
 * @file
 */
#include "bi/io/cpp/CppAliasGenerator.hpp"

#include "bi/io/cpp/CppBaseGenerator.hpp"

bi::CppAliasGenerator::CppAliasGenerator(std::ostream& base,
    const int level) :
    indentable_ostream(base, level) {
  //
}

void bi::CppAliasGenerator::visit(const Alias* o) {
  start("using ");
  CppBaseGenerator aux(base, level);
  aux << o->name << " = " << o->base;
  finish(';');
}
