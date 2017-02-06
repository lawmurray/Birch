/**
 * @file
 */
#include "bi/io/cpp/CppOutputGenerator.hpp"

#include "bi/visitor/Gatherer.hpp"

bi::CppOutputGenerator::CppOutputGenerator(std::ostream& base,
    const int level, const bool header) :
    CppBaseGenerator(base, level, header) {
  //
}

void bi::CppOutputGenerator::visit(const FuncParameter* o) {
  Gatherer<VarParameter> gatherer;
  o->result->accept(&gatherer);

  for (auto iter = gatherer.begin(); iter != gatherer.end(); ++iter) {
    line(*iter << ';');
  }
}
