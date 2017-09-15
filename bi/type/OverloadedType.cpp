/**
 * @file
 */
#include "bi/type/OverloadedType.hpp"

#include "bi/visitor/all.hpp"
#include "bi/exception/all.hpp"

bi::OverloadedType::OverloadedType(const poset<Type*,bi::definitely>& params,
    const std::map<Type*,Type*>& returns, Location* loc,
    const bool assignable) :
    Type(loc, assignable),
    params(params),
    returns(returns) {
  //
}

bi::OverloadedType::~OverloadedType() {
  //
}

bool bi::OverloadedType::isOverloaded() const {
  return true;
}

bi::Type* bi::OverloadedType::resolve(Type* args) {
  std::list<Type*> matches;
  params.match(args, matches);
  if (matches.size() == 1) {
    return returns[matches.front()];
  } else if (matches.size() == 0) {
    std::list<Type*> available;
    std::copy(params.begin(), params.end(), std::back_inserter(available));
    throw InvalidCallException(args, available);
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
