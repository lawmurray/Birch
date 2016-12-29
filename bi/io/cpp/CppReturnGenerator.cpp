/**
 * @file
 */
#include "bi/io/cpp/CppReturnGenerator.hpp"

bi::CppReturnGenerator::CppReturnGenerator(std::ostream& base,
    const int level, const bool header) :
    CppBaseGenerator(base, level, header) {
  //
}

void bi::CppReturnGenerator::visit(const VarParameter* o) {
  middle(o->name);
}

void bi::CppReturnGenerator::visit(const FuncParameter* o) {
  line("return " << o->result << ';');
}
