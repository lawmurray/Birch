/**
 * @file
 */
#include "bi/statement/AliasType.hpp"

#include "bi/visitor/all.hpp"

bi::AliasType::AliasType(shared_ptr<Name> name, Type* base,
    shared_ptr<Location> loc) :
    Statement(loc),
    Named(name),
    Based(base) {
  //
}

bi::AliasType::~AliasType() {
  //
}

bi::Statement* bi::AliasType::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::AliasType::accept(Modifier* visitor) {
  return visitor->modify(this);
}

bool bi::AliasType::dispatchDefinitely(const Statement& o) const {
  return o.definitely(*this);
}

bool bi::AliasType::definitely(const AliasType& o) const {
  return true;
}

bool bi::AliasType::dispatchPossibly(const Statement& o) const {
  return o.possibly(*this);
}

bool bi::AliasType::possibly(const AliasType& o) const {
  return true;
}
