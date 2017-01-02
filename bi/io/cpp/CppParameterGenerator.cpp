/**
 * @file
 */
#include "bi/io/cpp/CppParameterGenerator.hpp"

bi::CppParameterGenerator::CppParameterGenerator(std::ostream& base,
    const int level, const bool header) :
    CppBaseGenerator(base, level, header),
    n(1) {
  //
}

void bi::CppParameterGenerator::visit(const ModelReference* o) {
  if (!o->assignable) {
    middle("const ");
  }
  if (o->count() > 0) {
    middle("DefaultArray<" << o->name << "<HeapGroup>&," << o->count() << '>');
  } else {
    middle(o->name << "<>&");
  }
}

void bi::CppParameterGenerator::visit(const FuncParameter* o) {
  middle('(');
  for (auto iter = o->inputs.begin(); iter != o->inputs.end(); ++iter, ++n) {
    const VarParameter* param = *iter;
    if (iter != o->inputs.begin()) {
      middle(", ");
    }
    middle(param);
  }
  middle(')');
}
