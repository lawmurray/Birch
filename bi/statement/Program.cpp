/**
 * @file
 */
#include "bi/statement/Program.hpp"

#include "bi/visitor/all.hpp"

bi::Program::Program(Name* name, Expression* parens,
    Statement* braces, Location* loc) :
    Statement(loc),
    Named(name),
    Parenthesised(parens),
    Braced(braces) {
  //
}

bi::Program::~Program() {
  //
}

bi::Statement* bi::Program::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::Program::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Program::accept(Visitor* visitor) const {
  visitor->visit(this);
}
