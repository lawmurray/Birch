/**
 * @file
 */
#include "bi/expression/FuncReference.hpp"

#include "bi/visitor/all.hpp"

#include <vector>
#include <algorithm>
#include <typeinfo>

bi::FuncReference::FuncReference(shared_ptr<Name> name, Expression* parens,
    const SignatureForm form, shared_ptr<Location> loc, FuncParameter* target,
    Dispatcher* dispatcher) :
    Expression(loc),
    Named(name),
    Parenthesised(parens),
    Formed(form),
    Reference<FuncParameter>(target),
    dispatcher(dispatcher) {
  //
}

bi::FuncReference::FuncReference(Expression* left, shared_ptr<Name> name,
    Expression* right, const SignatureForm form, shared_ptr<Location> loc,
    FuncParameter* target, Dispatcher* dispatcher) :
    Expression(loc),
    Named(name),
    Parenthesised(new ParenthesesExpression(new ExpressionList(left, right))),
    Formed(form),
    Reference<FuncParameter>(target),
    dispatcher(dispatcher) {
  //
}

bi::FuncReference::~FuncReference() {
  //
}

bi::Expression* bi::FuncReference::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::FuncReference::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::FuncReference::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::FuncReference::dispatchDefinitely(Expression& o) {
  return o.definitely(*this);
}

bool bi::FuncReference::definitely(FuncReference& o) {
  return parens->definitely(*o.parens);
}

bool bi::FuncReference::definitely(FuncParameter& o) {
  return parens->definitely(*o.parens) && o.capture(this);
}

bool bi::FuncReference::definitely(Dispatcher& o) {
  auto f = [&](FuncParameter* o1) {
    return definitely(*o1);
  };
  auto iter = std::find_if(o.funcs.begin(), o.funcs.end(), f);
  return iter != o.funcs.end() && parens->definitely(*o.parens)
      && o.capture(this);
}

bool bi::FuncReference::definitely(VarParameter& o) {
  return type->definitely(*o.type) && o.capture(this);
}

bool bi::FuncReference::dispatchPossibly(Expression& o) {
  return o.possibly(*this);
}

bool bi::FuncReference::possibly(FuncReference& o) {
  return parens->possibly(*o.parens);
}

bool bi::FuncReference::possibly(FuncParameter& o) {
  return parens->possibly(*o.parens) && o.capture(this);
}

bool bi::FuncReference::possibly(Dispatcher& o) {
  auto f = [&](FuncParameter* o1) {
    return possibly(*o1);
  };
  auto iter = std::find_if(o.funcs.begin(), o.funcs.end(), f);
  return iter != o.funcs.end() && parens->possibly(*o.parens)
      && o.capture(this);
}

bool bi::FuncReference::possibly(VarParameter& o) {
  return type->possibly(*o.type) && o.capture(this);
}
