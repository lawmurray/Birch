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
  CppBaseGenerator::visit(o);
  if (!inDelay) {
    middle('&');
  }
}

void bi::CppParameterGenerator::visit(const BracketsType* o) {
  if (!o->assignable) {
    middle("const ");
  }
  CppBaseGenerator::visit(o);
  if (!inDelay) {
    middle('&');
  }
}

void bi::CppParameterGenerator::visit(const ParenthesesType* o) {
  if (dynamic_cast<TypeList*>(o->single->strip())) {
    if (!o->assignable) {
      middle("const ");
    }
    middle("std::tuple<" << o->single->strip() << ">");
    if (!inDelay) {
      middle('&');
    }
  } else {
    middle(o->single);
  }
}

void bi::CppParameterGenerator::visit(const DelayType* o) {
  if (!o->assignable) {
    middle("const ");
  }
  CppBaseGenerator::visit(o);
  if (!inDelay) {
    middle('&');
  }
}

void bi::CppParameterGenerator::visit(const LambdaType* o) {
  if (!o->assignable) {
    middle("const ");
  }
  CppBaseGenerator::visit(o);
  if (!inDelay) {
    middle('&');
  }
}

void bi::CppParameterGenerator::visit(const VarParameter* o) {
  middle(o->type << ' ' << o->name);
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

void bi::CppParameterGenerator::visit(const Dispatcher* o) {
  middle('(');
  for (auto iter = o->parens->begin(); iter != o->parens->end(); ++iter) {
    if (iter != o->parens->begin()) {
      middle(", ");
    }
    middle(*iter);
  }
  middle(')');
}
