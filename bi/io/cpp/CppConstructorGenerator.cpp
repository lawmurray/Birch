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
    start("bi::type::" << o->name);
    genTemplateSpec(o);
    middle("::");
  } else {
    start("");
  }
  middle(o->name);
  CppBaseGenerator aux(base, level, header);
  aux << '(' << o->params << ')';
  if (header) {
    finish(";\n");
  } else {
    finish(" :");
    in();
    in();
    start("super_type(");
    if (!o->args->isEmpty()) {
      middle(o->args);
    }
    middle(')');
    *this << o->braces->strip();
    out();
    out();
    finish(" {");
    in();
    line("//");
    out();
    line("}\n");
  }
}

void bi::CppConstructorGenerator::visit(const MemberVariable* o) {
  if (!o->value->isEmpty()) {
    finish(',');
    start(o->name << '(' << o->value << ')');
  } else if (o->type->isPointer() && !o->type->isWeak()) {
    finish(',');
    start(o->name << '(');
    middle("bi::make_pointer<" << o->type << '>');
    middle('(' << o->args << ')');
    middle(')');
  } else if (o->type->isArray() && !o->brackets->isEmpty()) {
    finish(',');
    start(o->name << "(bi::make_frame(" << o->brackets << ')');
    if (!o->args->isEmpty()) {
      middle(", " << o->args);
    }
    middle(')');
  }
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
