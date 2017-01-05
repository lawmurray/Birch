/**
 * @file
 */
#include "bi/expression/FuncParameter.hpp"

#include "bi/expression/FuncReference.hpp"
#include "bi/primitive/encode.hpp"
#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::FuncParameter::FuncParameter(shared_ptr<Name> name, Expression* parens,
    Expression* result, Expression* braces, const FunctionForm form, shared_ptr<Location> loc) :
    Expression(loc),
    Named(name),
    Braced(braces),
    Parenthesised(parens),
    Formed(form),
    result(result) {
  this->arg = this;
  if (parens->isRich()) {
    this->unique = new Name(uniqueName(this));
  } else {
    this->unique = new Name(internalName(this));
  }
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

bool bi::FuncParameter::dispatch(Expression& o) {
  return o.le(*this);
}

bool bi::FuncParameter::le(FuncParameter& o) {
  return *parens <= *o.parens && *type <= *o.type && o.capture(this);
}
