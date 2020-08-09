/**
 * @file
 */
#include "bi/statement/Block.hpp"

#include "bi/visitor/all.hpp"

bi::Block::Block(Statement* braces,Location* loc) :
    Statement(loc),
    Scoped(LOCAL_SCOPE),
    Braced(braces) {
  //
}

bi::Block::~Block() {
  //
}

bi::Statement* bi::Block::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::Block::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Block::accept(Visitor* visitor) const {
  visitor->visit(this);
}
