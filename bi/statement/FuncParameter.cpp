/**
 * @file
 */
#include "bi/statement/FuncParameter.hpp"

#include "bi/visitor/all.hpp"

bi::FuncParameter::FuncParameter(shared_ptr<Name> name, Expression* parens,
    Type* type, Expression* braces, const FunctionForm form,
    shared_ptr<Location> loc) :
    Statement(loc),
    Signature(name, parens, form),
    Typed(type),
    Braced(braces) {
  //
}

bi::FuncParameter::~FuncParameter() {
  //
}

bi::Statement* bi::FuncParameter::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::FuncParameter::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::FuncParameter::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::FuncParameter::dispatchDefinitely(const Statement& o) const {
  return o.definitely(*this);
}

bool bi::FuncParameter::definitely(const FuncParameter& o) const {
  return parens->definitely(*o.parens);
}

bool bi::FuncParameter::dispatchPossibly(const Statement& o) const {
  return o.possibly(*this);
}

bool bi::FuncParameter::possibly(const FuncParameter& o) const {
  return parens->possibly(*o.parens);
}
