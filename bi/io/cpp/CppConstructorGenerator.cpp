/**
 * @file
 */
#include "bi/io/cpp/CppConstructorGenerator.hpp"

#include "bi/io/cpp/CppParameterGenerator.hpp"

bi::CppConstructorGenerator::CppConstructorGenerator(std::ostream& base,
    const int level, const bool header) :
    CppBaseGenerator(base, level, header),
    before(false) {
  //
}

void bi::CppConstructorGenerator::visit(const TypeParameter* o) {
  if (header) {
    start(o->name);

    CppParameterGenerator auxParameter(base, level, header);
    auxParameter << o;

    finish(" :");
    in();
    in();
    if (!o->base->isEmpty()) {
      before = true;
      start("super_type(");
      if (!o->baseParens->isEmpty()) {
        middle(o->baseParens);
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
  if (!o->parens->isEmpty()) {
    if (o->type->isArray()) {
      middle(", ");
    }
    middle(o->parens);
  }
  if (!o->value->isEmpty()) {
    if (o->type->isArray() || !o->parens->isEmpty()) {
      middle(", ");
    }
    middle(o->value);
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
