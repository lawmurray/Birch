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
    Scoped(LOCAL_SCOPE),
    Braced(braces) {
  //
}

birch::Program::~Program() {
  //
}

bool birch::Program::isDeclaration() const {
  return true;
}

birch::Statement* birch::Program::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

birch::Statement* birch::Program::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::Program::accept(Visitor* visitor) const {
  visitor->visit(this);
}
