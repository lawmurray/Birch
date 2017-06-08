/**
 * @file
 */
#include "bi/expression/FuncParameter.hpp"

#include "bi/visitor/all.hpp"

bi::FuncParameter::FuncParameter(shared_ptr<Name> name, Expression* parens,
    Type* type, Expression* braces, const FunctionForm form,
    shared_ptr<Location> loc) :
    Expression(type, loc),
    Signature(name, parens, form),
    Braced(braces) {
  //
}

bi::FuncParameter::FuncParameter(Expression* left, shared_ptr<Name> name,
    Expression* right, Type* type, Expression* braces,
    const FunctionForm form, shared_ptr<Location> loc) :
    Expression(type, loc),
    Signature(name, new ExpressionList(left, right), form),
    Braced(braces) {
  //
}

bi::FuncParameter::~FuncParameter() {
  //
}

bi::Expression* bi::FuncParameter::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::FuncParameter::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::FuncParameter::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::FuncParameter::dispatchDefinitely(const Expression& o) const {
  return o.definitely(*this);
}

bool bi::FuncParameter::definitely(const FuncParameter& o) const {
  return parens->definitely(*o.parens);
}

bool bi::FuncParameter::dispatchPossibly(const Expression& o) const {
  return o.possibly(*this);
}

bool bi::FuncParameter::possibly(const FuncParameter& o) const {
  return parens->possibly(*o.parens);
}
