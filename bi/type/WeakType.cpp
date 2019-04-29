/**
 * @file
 */
#include "bi/type/WeakType.hpp"

#include "bi/visitor/all.hpp"

bi::WeakType::WeakType(Type* single, Location* loc) :
    Type(loc),
    Single<Type>(single) {
  //
}

bi::WeakType::~WeakType() {
  //
}

bi::Type* bi::WeakType::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Type* bi::WeakType::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::WeakType::accept(Visitor* visitor) const {
  return visitor->visit(this);
}

bool bi::WeakType::isWeak() const {
  return true;
}

const bi::Type* bi::WeakType::unwrap() const {
  return single;
}

bi::Type* bi::WeakType::unwrap() {
  return single;
}

void bi::WeakType::resolveConstructor(Argumented* o) {
  if (!o->args->isEmpty()) {
    throw ConstructorException(o);
  }
}

bool bi::WeakType::dispatchIsConvertible(const Type& o) const {
  return o.isConvertible(*this);
}

bool bi::WeakType::isConvertible(const GenericType& o) const {
  assert(o.target);
  return isConvertible(*o.target->type);
}

bool bi::WeakType::isConvertible(const MemberType& o) const {
  return isConvertible(*o.right);
}

bool bi::WeakType::isConvertible(const OptionalType& o) const {
  if (o.single->isWeak()) {
    return isConvertible(*o.single);
  } else {
    /* a weak pointer can assign to an optional of a shared pointer */
    return o.single->isConvertible(*o.single);
  }
}

bool bi::WeakType::isConvertible(const WeakType& o) const {
  return single->isConvertible(*o.single);
}

bool bi::WeakType::dispatchIsAssignable(const Type& o) const {
  return o.isAssignable(*this);
}

bool bi::WeakType::isAssignable(const ClassType& o) const {
  return o.getClass()->hasAssignment(this);
}

bool bi::WeakType::isAssignable(const GenericType& o) const {
  assert(o.target);
  return isAssignable(*o.target->type);
}

bool bi::WeakType::isAssignable(const MemberType& o) const {
  return isAssignable(*o.right);
}

bool bi::WeakType::isAssignable(const OptionalType& o) const {
  if (o.single->isWeak()) {
    return isAssignable(*o.single);
  } else {
    /* a weak pointer can assign to an optional of a shared pointer */
    return o.single->isAssignable(*o.single);
  }
}

bool bi::WeakType::isAssignable(const WeakType& o) const {
  return single->isAssignable(*o.single);
}

bi::Type* bi::WeakType::dispatchCommon(const Type& o) const {
  return o.common(*this);
}

bi::Type* bi::WeakType::common(const ClassType& o) const {
  auto single1 = single->common(o);
  if (single1) {
    return new WeakType(single1);
  } else {
    return nullptr;
  }
}

bi::Type* bi::WeakType::common(const GenericType& o) const {
  assert(o.target);
  return common(*o.target->type);
}

bi::Type* bi::WeakType::common(const MemberType& o) const {
  return common(*o.right);
}

bi::Type* bi::WeakType::common(const OptionalType& o) const {
  if (o.single->isWeak()) {
    auto single1 = common(*o.single);
    if (single1) {
      return new OptionalType(single1);
    }
  } else {
    auto single1 = single->common(*o.single);
    if (single1) {
      return new OptionalType(single1);
    }
  }
  return nullptr;
}

bi::Type* bi::WeakType::common(const WeakType& o) const {
  auto single1 = single->common(*o.single);
  if (single1) {
    return new WeakType(single1);
  } else {
    return nullptr;
  }
}
