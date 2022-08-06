/**
 * @file
 */
#include "src/expression/Assign.hpp"

#include "src/visitor/all.hpp"

birch::Assign::Assign(Expression* left, Name* op, Expression* right,
    Location* loc) :
    Expression(loc),
    Named(op),
    Couple<Expression>(left, right) {
  //
}

void birch::Assign::accept(Visitor* visitor) const {
  visitor->visit(this);
}
