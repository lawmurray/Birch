/**
 * @file
 */
#include "bi/expression/FuncParameter.hpp"

#include "bi/visitor/all.hpp"

#include <algorithm>

bi::FuncParameter::FuncParameter(shared_ptr<Name> name, Expression* parens,
    Expression* result, Expression* braces, const SignatureForm form,
    shared_ptr<Location> loc) :
    Expression(loc),
    Signature(name, parens, result, form),
    Braced(braces) {
  this->arg = this;
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

bool bi::FuncParameter::definitely(Dispatcher& o) {
  auto f = [&](FuncParameter* o1) {
    return definitely(*o1);
  };
  auto iter = std::find_if(o.funcs.begin(), o.funcs.end(), f);
  return iter != o.funcs.end() && o.capture(this);
}

bool bi::FuncParameter::dispatchPossibly(Expression& o) {
  return o.possibly(*this);
}

bool bi::FuncParameter::possibly(FuncParameter& o) {
  return parens->possibly(*o.parens) && o.capture(this);
}

bool bi::FuncParameter::possibly(Dispatcher& o) {
  auto f = [&](FuncParameter* o1) {
    return possibly(*o1);
  };
  auto iter = std::find_if(o.funcs.begin(), o.funcs.end(), f);
  return iter != o.funcs.end() && o.capture(this);
}
