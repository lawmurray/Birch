/**
 * @file
 */
#include "bi/statement/Program.hpp"

#include "bi/visitor/all.hpp"

bi::Program::Program(shared_ptr<Name> name, Expression* parens,
    Expression* braces, shared_ptr<Location> loc) :
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

bool bi::Program::dispatchDefinitely(const Statement& o) const {
  return o.definitely(*this);
}

bool bi::Program::definitely(const Program& o) const {
  return parens->definitely(*o.parens) && braces->definitely(*o.braces);
}

bool bi::Program::dispatchPossibly(const Statement& o) const {
  return o.possibly(*this);
}

bool bi::Program::possibly(const Program& o) const {
  return parens->possibly(*o.parens) && braces->possibly(*o.braces);
}
