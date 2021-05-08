/**
 * @file
 */
#include "src/statement/Program.hpp"

#include "src/visitor/all.hpp"

birch::Program::Program(Name* name, Expression* params, Statement* braces,
    Location* loc) :
    Statement(loc),
    Named(name),
    Parameterised(params),
    Braced(braces) {
  //
}

void birch::Program::accept(Visitor* visitor) const {
  visitor->visit(this);
}
