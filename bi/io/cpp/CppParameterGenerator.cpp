/**
 * @file
 */
#include "bi/io/cpp/CppParameterGenerator.hpp"

bi::CppParameterGenerator::CppParameterGenerator(std::ostream& base,
    const int level, const bool header) :
    CppBaseGenerator(base, level, header) {
  //
}

void bi::CppParameterGenerator::visit(const VarParameter* o) {
  if (!o->type->assignable) {
    middle("const ");
  }
  middle(o->type << "& " << o->name);
  if (!o->value->isEmpty()) {
    middle(" = " << o->value);
  }
}

void bi::CppParameterGenerator::visit(const FuncParameter* o) {
  middle('(');
  for (auto iter = o->parens->begin(); iter != o->parens->end(); ++iter) {
    if (iter != o->parens->begin()) {
      middle(", ");
    }
    middle(*iter);
  }
  middle(')');
}
