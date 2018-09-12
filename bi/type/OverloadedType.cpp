/**
 * @file
 */
#include "bi/type/OverloadedType.hpp"

#include "bi/visitor/all.hpp"
#include "bi/exception/all.hpp"

bi::OverloadedType::OverloadedType(Overloaded* overloaded, Location* loc) :
    Type(loc),
    overloaded(overloaded) {
  //
}

bi::OverloadedType::~OverloadedType() {
  //
}

bool bi::OverloadedType::isOverloaded() const {
  return true;
}

bi::FunctionType* bi::OverloadedType::resolve(Argumented* o) {
  std::set<Parameterised*> matches;
  overloaded->overloads.match(o, matches);
  if (matches.size() == 1) {
    /* construct the appropriate function type */
    auto target = *matches.begin();
    Type* paramsType = target->params->type;
    Type* returnType = dynamic_cast<ReturnTyped*>(target)->returnType;
    return new FunctionType(paramsType, returnType);
  } else if (matches.size() == 0) {
    std::list<Parameterised*> available;
    std::copy(overloaded->overloads.begin(), overloaded->overloads.end(),
        std::back_inserter(available));
    throw CallException(o, available);
  } else {
    throw AmbiguousCallException(o, matches);
  }
}

bi::FunctionType* bi::OverloadedType::resolve() const {
  assert(overloaded->size() == 1);
  auto target = overloaded->front();
  Type* paramsType = target->params->type;
  Type* returnType = dynamic_cast<ReturnTyped*>(target)->returnType;
  return new FunctionType(paramsType, returnType);
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

bool bi::OverloadedType::definitely(const FunctionType& o) const {
  if (overloaded->size() == 1) {
    return resolve()->definitely(o);
  } else {
    return false;
  }
}

bi::Type* bi::OverloadedType::dispatchCommon(const Type& o) const {
  return o.common(*this);
}

bi::Type* bi::OverloadedType::common(const OverloadedType& o) const {
  return nullptr;
}

bi::Type* bi::OverloadedType::common(const FunctionType& o) const {
  if (overloaded->size() == 1) {
    return resolve()->common(o);
  } else {
    return nullptr;
  }
}
