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

void bi::CppConstructorGenerator::visit(const TypeParameter* o) {
  if (header) {
    start(o->name << '(');
    if (!o->parens->isEmpty()) {
      middle(o->parens);
    }
    middle(')');
    finish(" :");
    in();
    in();
    if (!o->base->isEmpty()) {
      before = true;
      start("base_type(");
      if (o->super() && !o->super()->parens->isEmpty()) {
        middle(o->super()->parens);
      }
      middle(')');
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

void bi::CppConstructorGenerator::visit(const VarParameter* o) {
  if (before) {
    finish(',');
  }
  before = true;

  start(o->name << '(');
  if (o->type->isArray()) {
    const BracketsType* type =
        dynamic_cast<const BracketsType*>(o->type.get());
    assert(type);
    middle("make_frame(" << type->brackets << ")");
  }
  middle(')');
}

void bi::CppConstructorGenerator::visit(const VarDeclaration* o) {
  *this << o->param;
}

void bi::CppConstructorGenerator::visit(const FuncDeclaration* o) {
  //
}

void bi::CppConstructorGenerator::visit(const ConversionDeclaration* o) {
  //
}
