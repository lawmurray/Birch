/**
 * @file
 */
#include "bi/statement/Conditional.hpp"

#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::Conditional::Conditional(Expression* cond, Expression* braces,
    Expression* falseBraces, shared_ptr<Location> loc) :
    Statement(loc), Conditioned(cond), Braced(braces), falseBraces(
        falseBraces) {
  /* pre-condition */
  assert(falseBraces);
}

bi::Statement* bi::Conditional::acceptClone(Cloner* visitor) const {
  return visitor->clone(this);
}

void bi::Conditional::acceptModify(Modifier* visitor) {
  visitor->modify(this);
}

void bi::Conditional::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::Conditional::operator<=(Statement& o) {
  try {
    Conditional& o1 = dynamic_cast<Conditional&>(o);
    return *cond <= *o1.cond && *braces <= *o1.braces
        && *falseBraces <= *o1.falseBraces;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}

bool bi::Conditional::operator==(const Statement& o) const {
  try {
    const Conditional& o1 = dynamic_cast<const Conditional&>(o);
    return *cond == *o1.cond && *braces == *o1.braces
        && *falseBraces == *o1.falseBraces;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}
