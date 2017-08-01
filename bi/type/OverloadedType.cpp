/**
 * @file
 */
#include "bi/type/OverloadedType.hpp"

#include "bi/visitor/all.hpp"
#include "bi/exception/all.hpp"

bi::OverloadedType::OverloadedType(
    const poset<Type*,bi::definitely>& overloads, Location* loc,
    const bool assignable) :
    Type(loc, assignable),
    overloads(overloads) {
  //
}

bi::OverloadedType::~OverloadedType() {
  //
}

bool bi::OverloadedType::contains(Type* o) const {
  return overloads.contains(o);
}

void bi::OverloadedType::add(Type* o) {
  /* pre-condition */
  assert(!contains(o));

  overloads.insert(o);
}

bool bi::OverloadedType::isOverloaded() const {
  return true;
}

bi::Type* bi::OverloadedType::resolve(Type* args) {
  std::list<Type*> matches;
  overloads.match(args, matches);
  if (matches.size() == 1) {
    return matches.front()->resolve(args);
  } else if (matches.size() == 0) {
    throw InvalidCallException(args);
  } else {
    throw AmbiguousCallException(args, matches);
  }
}

bi::Type* bi::OverloadedType::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Type* bi::OverloadedType::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::OverloadedType::accept(Visitor* visitor) const {
  return visitor->visit(this);
}

bool bi::OverloadedType::dispatchDefinitely(const Type& o) const {
  return o.definitely(*this);
}

bool bi::OverloadedType::definitely(const OverloadedType& o) const {
  return false;
}

bool bi::OverloadedType::dispatchPossibly(const Type& o) const {
  return o.possibly(*this);
}

bool bi::OverloadedType::possibly(const OverloadedType& o) const {
  return false;
}
