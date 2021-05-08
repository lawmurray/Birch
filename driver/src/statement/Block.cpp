/**
 * @file
 */
#include "src/statement/Block.hpp"

#include "src/visitor/all.hpp"

birch::Block::Block(Statement* braces,Location* loc) :
    Statement(loc),
    Braced(braces) {
  //
}

void birch::Block::accept(Visitor* visitor) const {
  visitor->visit(this);
}
