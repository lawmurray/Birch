/**
 * @file
 */
#include "bi/io/cpp/CppParameterGenerator.hpp"

bi::CppParameterGenerator::CppParameterGenerator(std::ostream& base,
    const int level, const bool header) :
    CppBaseGenerator(base, level, header) {
  //
}

void bi::CppParameterGenerator::visit(const ModelReference* o) {
  if (!o->assignable) {
    middle("const ");
  }
  if (o->count() > 0) {
    middle("DefaultArray<" << o->name << "<HeapGroup>," << o->count() << '>');
  } else {
    middle("bi::model::" << o->name << "<>");
  }
}

void bi::CppParameterGenerator::visit(const VarParameter* o) {
  middle(o->type << "& " << o->name);
  if (!o->value->isEmpty()) {
    middle(" = " << o->value);
  }
}

void bi::CppParameterGenerator::visit(const FuncParameter* o) {
  middle('(');
  for (auto iter = o->inputs.begin(); iter != o->inputs.end(); ++iter) {
    if (iter != o->inputs.begin()) {
      middle(", ");
    }
    middle(*iter);
  }
  middle(')');
}
