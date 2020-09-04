/**
 * @file
 */
#include "src/statement/While.hpp"

#include "src/visitor/all.hpp"

birch::While::While(Expression* cond, Statement* braces, Location* loc) :
    Statement(loc),
    Conditioned(cond),
    Scoped(LOCAL_SCOPE),
    Braced(braces) {
  //
}

birch::While::~While() {
  //
}

birch::Statement* birch::While::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

birch::Statement* birch::While::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::While::accept(Visitor* visitor) const {
  visitor->visit(this);
}
