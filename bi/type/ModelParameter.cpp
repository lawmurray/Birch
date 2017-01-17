/**
 * @file
 */
#include "bi/type/ModelParameter.hpp"

#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::ModelParameter::ModelParameter(shared_ptr<Name> name, Expression* parens,
    shared_ptr<Name> op, Type* base, Expression* braces,
    shared_ptr<Location> loc) :
    Type(loc),
    Named(name),
    Parenthesised(parens),
    Based(op, base),
    Braced(braces) {
  this->arg = this;
}

bi::ModelParameter::~ModelParameter() {
  //
}

const std::list<bi::VarParameter*>& bi::ModelParameter::vars() const {
  /* pre-condition */
  assert(scope);

  return scope->vars.ordered;
}

const std::list<bi::FuncParameter*>& bi::ModelParameter::funcs() const {
  /* pre-condition */
  assert(scope);

  return scope->funcs.ordered;
}

bi::Type* bi::ModelParameter::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Type* bi::ModelParameter::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::ModelParameter::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::ModelParameter::builtin() const {
  if (*op == "=") {
    return base->builtin();
  } else {
    return braces->isEmpty();
  }
}

bi::possibly bi::ModelParameter::dispatch(Type& o) {
  return o.le(*this);
}

bi::possibly bi::ModelParameter::le(ModelParameter& o) {
  return *parens <= *o.parens && *base <= *o.base && o.capture(this);
}

bi::possibly bi::ModelParameter::le(EmptyType& o) {
  return definite;
}
