/**
 * @file
 */
#include "bi/expression/FuncParameter.hpp"

#include "bi/primitive/encode.hpp"
#include "bi/visitor/all.hpp"

bi::FuncParameter::FuncParameter(shared_ptr<Name> name, Expression* parens,
    Expression* result, Expression* braces, const FunctionForm form, shared_ptr<Location> loc) :
    Expression(loc),
    Named(name),
    Formed(parens, form),
    Braced(braces),
    result(result) {
  this->arg = this;
  this->mangled = new Name(mangle(this));
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

bool bi::FuncParameter::dispatchDefinitely(Expression& o) {
  return o.definitely(*this);
}

bool bi::FuncParameter::definitely(FuncParameter& o) {
  return parens->definitely(*o.parens) && o.capture(this);
}

bool bi::FuncParameter::dispatchPossibly(Expression& o) {
  return o.possibly(*this);
}

bool bi::FuncParameter::possibly(FuncParameter& o) {
  return parens->possibly(*o.parens) && o.capture(this);
}
