/**
 * @file
 */
#include "bi/io/cpp/CppConstructorGenerator.hpp"

bi::CppConstructorGenerator::CppConstructorGenerator(std::ostream& base,
    const int level, const bool header) :
    CppBaseGenerator(base, level, header) {
  //
}

void bi::CppConstructorGenerator::visit(const Class* o) {
  if (!header) {
    start("bi::" << o->name << "::");
  } else {
    start("");
  }
  middle(o->name);
  if (o->parens->isEmpty()) {
    middle("()");
  } else {
    CppBaseGenerator aux(base, level, header);
    aux << o->parens;
  }
  if (header) {
    finish(";\n");
  } else {
    finish(" :");
    in();
    in();
    start("super_type");
    if (!o->baseParens->isEmpty()) {
      middle(o->baseParens);
    } else {
      middle("()");
    }
    for (auto iter = o->parens->begin(); iter != o->parens->end(); ++iter) {
      *this << *iter;
    }
    *this << o->braces;
    out();
    out();
    finish(" {");
    in();
    line("//");
    out();
    line("}\n");
  }
}

void bi::CppConstructorGenerator::visit(const MemberParameter* o) {
  finish(',');
  start(o->name << '(' << o->name << ')');
}

void bi::CppConstructorGenerator::visit(const MemberVariable* o) {
  finish(',');
  start(o->name << '(');
  if (o->type->isClass()) {
    ClassType* type = dynamic_cast<ClassType*>(o->type);
    assert(type);
    middle("bi::make_object<" << type->name << '>');
    if (o->parens->isEmpty()) {
      middle("()");
    } else {
      middle(o->parens);
    }
  } else if (o->type->isArray()) {
    const ArrayType* type = dynamic_cast<const ArrayType*>(o->type);
    assert(type);
    middle("bi::make_frame(" << type->brackets << ")");
    if (!o->parens->isEmpty()) {
      middle(", " << o->parens->strip());
    }
  } else if (!o->value->isEmpty()) {
    middle(o->value);
  }
  middle(')');
}

void bi::CppConstructorGenerator::visit(const MemberFunction* o) {
  //
}

void bi::CppConstructorGenerator::visit(const MemberFiber* o) {
  //
}

void bi::CppConstructorGenerator::visit(const ConversionOperator* o) {
  //
}

void bi::CppConstructorGenerator::visit(const AssignmentOperator* o) {
  //
}
