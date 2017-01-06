/**
 * @file
 */
#include "bi/expression/FuncReference.hpp"

#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::FuncReference::FuncReference(shared_ptr<Name> name, Expression* parens,
    const FunctionForm form, shared_ptr<Location> loc,
    const FuncParameter* target) :
    Expression(loc),
    Named(name),
    Reference<FuncParameter>(target),
    Parenthesised(parens),
    Formed(form) {
  //
}

bi::FuncReference::FuncReference(Expression* left, shared_ptr<Name> name,
    Expression* right, shared_ptr<Location> loc, const FuncParameter* target) :
    Expression(loc),
    Named(name),
    Reference<FuncParameter>(target),
    Parenthesised(new ParenthesesExpression(new ExpressionList(left, right))),
    Formed(BINARY_OPERATOR) {
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

bool bi::FuncReference::dispatch(Expression& o) {
  return o.le(*this);
}

bool bi::FuncReference::le(FuncReference& o) {
  return *parens <= *o.parens && target == o.target;
}

bool bi::FuncReference::le(FuncParameter& o) {
  if (!target) {
    /* not yet bound */
    return *parens <= *o.parens && o.capture(this);
  } else {
    return *parens <= *o.parens && *type <= *o.type && o.capture(this);
  }
}

bool bi::FuncReference::le(VarParameter& o) {
  return *type <= *o.type && o.capture(this);
}
