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
  if (!o->result->isEmpty()) {
    for (auto iter = o->outputs.begin(); iter != o->outputs.end(); ++iter) {
      line(*iter << ';');
    }
  }
}
