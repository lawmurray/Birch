/**
 * @file
 */
#include "bi/statement/Alias.hpp"

#include "bi/visitor/all.hpp"

bi::Alias::Alias(shared_ptr<Name> name, Type* base, shared_ptr<Location> loc) :
    Statement(loc),
    Named(name),
    Based(base) {
  //
}

bi::Alias::~Alias() {
  //
}

const bi::Statement* bi::Alias::super() const {
  auto base = dynamic_cast<const Identifier<Class>*>(this->base.get());
  assert(base);
  return base->target;
}

const bi::Statement* bi::Alias::canonical() const {
  return super();
}

bi::Statement* bi::Alias::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::Alias::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Alias::accept(Visitor* visitor) const {
  return visitor->visit(this);
}

bool bi::Alias::dispatchDefinitely(const Statement& o) const {
  return o.definitely(*this);
}

bool bi::Alias::definitely(const Alias& o) const {
  return true;
}

bool bi::Alias::dispatchPossibly(const Statement& o) const {
  return o.possibly(*this);
}

bool bi::Alias::possibly(const Alias& o) const {
  return true;
}
