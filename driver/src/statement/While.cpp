/**
 * @file
 */
#include "src/statement/While.hpp"

#include "src/visitor/all.hpp"

birch::While::While(Expression* cond, Statement* braces, Location* loc) :
    Statement(loc),
    Conditioned(cond),
    Braced(braces) {
  //
}

void birch::While::accept(Visitor* visitor) const {
  visitor->visit(this);
}
