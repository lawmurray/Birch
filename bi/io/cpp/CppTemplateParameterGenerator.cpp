/**
 * @file
 */
#include "bi/io/cpp/CppTemplateParameterGenerator.hpp"

#include "bi/visitor/Gatherer.hpp"

bi::CppTemplateParameterGenerator::CppTemplateParameterGenerator(std::ostream& base,
    const int level, const bool header) :
    CppBaseGenerator(base, level, header), args(0) {
  //
}

void bi::CppTemplateParameterGenerator::visit(const VarParameter* o) {
  if (o->type->assignable) {
    if (args == 0) {
      start("template<");
    } else {
      middle(", ");
    }
    middle("class Arg" << ++args << '_');
  }
}

void bi::CppTemplateParameterGenerator::visit(const FuncParameter* o) {
  Gatherer<VarParameter> gatherer;
  o->parens->accept(&gatherer);

  for (auto iter = gatherer.begin(); iter != gatherer.end(); ++iter) {
    middle(*iter);
  }
  if (args > 0) {
    finish('>');
  }
}

void bi::CppTemplateParameterGenerator::visit(const Dispatcher* o) {
  Gatherer<VarParameter> gatherer;
  o->parens->accept(&gatherer);

  for (auto iter = gatherer.begin(); iter != gatherer.end(); ++iter) {
    middle(*iter);
  }
  if (args > 0) {
    finish('>');
  }
}
