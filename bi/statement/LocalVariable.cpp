/**
 * @file
 */
#include "bi/statement/LocalVariable.hpp"

#include "bi/visitor/all.hpp"

bi::LocalVariable::LocalVariable(shared_ptr<Name> name, Type* type,
    Expression* parens, Expression* value, shared_ptr<Location> loc) :
    Statement(loc),
    Named(name),
    Typed(type),
    Parenthesised(parens),
    Valued(value) {
  //
}

bi::LocalVariable::~LocalVariable() {
  //
}

bi::Statement* bi::LocalVariable::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::LocalVariable::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::LocalVariable::accept(Visitor* visitor) const {
  visitor->visit(this);
}
