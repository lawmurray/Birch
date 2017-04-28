/**
 * @file
 */
#include "bi/statement/For.hpp"

#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::For::For(Expression* index, Expression* from, Expression* to,
    Expression* braces, shared_ptr<Location> loc) :
    Statement(loc),
    Braced(braces),
    index(index),
    from(from),
    to(to) {
  //
}

bi::For::~For() {
  //
}

bi::Statement* bi::For::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::For::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::For::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::For::dispatchDefinitely(const Statement& o) const {
  return o.definitely(*this);
}

bool bi::For::definitely(const For& o) const {
  return index->definitely(*o.index) && from->definitely(*o.from)
      && to->definitely(*o.to);
}

bool bi::For::dispatchPossibly(const Statement& o) const {
  return o.possibly(*this);
}

bool bi::For::possibly(const For& o) const {
  return index->possibly(*o.index) && from->possibly(*o.from)
      && to->possibly(*o.to);
}
