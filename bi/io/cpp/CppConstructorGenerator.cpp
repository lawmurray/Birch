/**
 * @file
 */
#include "bi/io/cpp/CppConstructorGenerator.hpp"

bi::CppConstructorGenerator::CppConstructorGenerator(std::ostream& base,
    const int level, const bool header) :
    CppBaseGenerator(base, level, header),
    before(false) {
  //
}

void bi::CppConstructorGenerator::visit(const Class* o) {
  if (!header) {
    start("bi::type::" << o->name << "::");
  } else {
    start("");
  }
  middle(o->name << '(');
  CppBaseGenerator aux(base, level, header);
  aux << o->parens;
  middle(')');
  if (header) {
    finish(";\n");
  } else {
    finish(" :");
    in();
    in();
    if (!o->base->isEmpty()) {
      before = true;
      start("super_type(" << o->baseParens << ')');
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
  if (before) {
    finish(',');
  }
  before = true;
  start(o->name << '(' << o->name << ')');
}

void bi::CppConstructorGenerator::visit(const MemberVariable* o) {
  if (before) {
    finish(',');
  }
  before = true;

  start(o->name << '(');
  if (o->type->isArray()) {
    const ArrayType* type =
        dynamic_cast<const ArrayType*>(o->type.get());
    assert(type);
    middle("make_frame(" << type->brackets << ")");
  } else if (!o->value->isEmpty()) {
    middle(o->value);
  }
  middle(')');
}

void bi::CppConstructorGenerator::visit(const MemberFunction* o) {
  //
}

void bi::CppConstructorGenerator::visit(const ConversionOperator* o) {
  //
}

void bi::CppConstructorGenerator::visit(const AssignmentOperator* o) {
  //
}
