/**
 * @file
 */
#include "bi/io/cpp/CppOutputGenerator.hpp"

bi::CppOutputGenerator::CppOutputGenerator(std::ostream& base,
    const int level, const bool header) :
    CppBaseGenerator(base, level, header) {
  //
}

void bi::CppOutputGenerator::visit(const FuncParameter* o) {
  for (auto iter = o->result->begin(); iter != o->result->end(); ++iter) {
    line(*iter << ';');
  }
}
