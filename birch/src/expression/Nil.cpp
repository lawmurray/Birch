/**
 * @file
 */
#include "src/expression/Nil.hpp"

#include "src/visitor/all.hpp"

birch::Nil::Nil(Location* loc) :
    Expression(loc) {
  //
}

void birch::Nil::accept(Visitor* visitor) const {
  visitor->visit(this);
}
