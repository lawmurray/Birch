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

bi::Type* bi::ModelParameter::acceptClone(Cloner* visitor) const {
  return visitor->clone(this);
}

void bi::ModelParameter::acceptModify(Modifier* visitor) {
  visitor->modify(this);
}

void bi::ModelParameter::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::ModelParameter::builtin() const {
  if (*op == "=") {
    return base->builtin();
  } else {
    return !*braces;
  }
}

bool bi::ModelParameter::operator<=(Type& o) {
  try {
    ModelParameter& o1 = dynamic_cast<ModelParameter&>(o);
    return *parens <= *o1.parens && *base <= *o1.base && o1.capture(this);
  } catch (std::bad_cast e) {
    //
  }
  try {
    ParenthesesType& o1 = dynamic_cast<ParenthesesType&>(o);
    return *this <= *o1.type;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}

bool bi::ModelParameter::operator==(const Type& o) const {
  try {
    const ModelParameter& o1 = dynamic_cast<const ModelParameter&>(o);
    return *parens == *o1.parens && *base == *o1.base;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}
