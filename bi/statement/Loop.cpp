/**
 * @file
 */
#include "bi/statement/Loop.hpp"

#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::Loop::Loop(Expression* cond, Expression* braces, shared_ptr<Location> loc) :
    Statement(loc), Conditioned(cond), Braced(braces) {
  //
}

bi::Loop::~Loop() {
  //
}

bi::Statement* bi::Loop::acceptClone(Cloner* visitor) const {
  return visitor->clone(this);
}

void bi::Loop::acceptModify(Modifier* visitor) {
  visitor->modify(this);
}

void bi::Loop::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::Loop::operator<=(Statement& o) {
  try {
    Loop& o1 = dynamic_cast<Loop&>(o);
    return *cond <= *o1.cond && *braces <= *o1.braces;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}

bool bi::Loop::operator==(const Statement& o) const {
  try {
    const Loop& o1 = dynamic_cast<const Loop&>(o);
    return *cond == *o1.cond && *braces == *o1.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}
