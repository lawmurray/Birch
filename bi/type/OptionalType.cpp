/**
 * @file
 */
#include "bi/type/OptionalType.hpp"

#include "bi/visitor/all.hpp"

bi::OptionalType::OptionalType(Type* single, Location* loc) :
    Type(loc),
    Single<Type>(single) {
  //
}

bi::OptionalType::~OptionalType() {
  //
}

bi::Type* bi::OptionalType::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Type* bi::OptionalType::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::OptionalType::accept(Visitor* visitor) const {
  return visitor->visit(this);
}

bool bi::OptionalType::isValue() const {
  return single->isValue();
}

bool bi::OptionalType::isOptional() const {
  return true;
}

const bi::Type* bi::OptionalType::unwrap() const {
  return single;
}

bi::Type* bi::OptionalType::unwrap() {
  return single;
}

bool bi::OptionalType::dispatchIsConvertible(const Type& o) const {
  return o.isConvertible(*this);
}

bool bi::OptionalType::isConvertible(const GenericType& o) const {
  assert(o.target);
  return isConvertible(*o.target->type);
}

bool bi::OptionalType::isConvertible(const MemberType& o) const {
  return isConvertible(*o.right);
}

bool bi::OptionalType::isConvertible(const OptionalType& o) const {
  return single->isConvertible(*o.single);
}

bool bi::OptionalType::isConvertible(const WeakType& o) const {
  return single->isConvertible(*o.single);
}

bi::Type* bi::OptionalType::dispatchCommon(const Type& o) const {
  return o.common(*this);
}

bi::Type* bi::OptionalType::common(const ArrayType& o) const {
  auto single1 = single->common(o);
  if (single1) {
    return new OptionalType(single1);
  } else {
    return nullptr;
  }
}

bi::Type* bi::OptionalType::common(const BasicType& o) const {
  auto single1 = single->common(o);
  if (single1) {
    return new OptionalType(single1);
  } else {
    return nullptr;
  }
}

bi::Type* bi::OptionalType::common(const BinaryType& o) const {
  auto single1 = single->common(o);
  if (single1) {
    return new OptionalType(single1);
  } else {
    return nullptr;
  }
}

bi::Type* bi::OptionalType::common(const ClassType& o) const {
  auto single1 = single->common(o);
  if (single1) {
    return new OptionalType(single1);
  } else {
    return nullptr;
  }
}

bi::Type* bi::OptionalType::common(const FiberType& o) const {
  auto single1 = single->common(o);
  if (single1) {
    return new OptionalType(single1);
  } else {
    return nullptr;
  }
}

bi::Type* bi::OptionalType::common(const FunctionType& o) const {
  auto single1 = single->common(o);
  if (single1) {
    return new OptionalType(single1);
  } else {
    return nullptr;
  }
}

bi::Type* bi::OptionalType::common(const GenericType& o) const {
  assert(o.target);
  return common(*o.target->type);
}

bi::Type* bi::OptionalType::common(const MemberType& o) const {
  return common(*o.right);
}

bi::Type* bi::OptionalType::common(const NilType& o) const {
  return new OptionalType(single->common(*single));
}

bi::Type* bi::OptionalType::common(const OptionalType& o) const {
  auto single1 = single->common(*o.single);
  if (single1) {
    return new OptionalType(single1);
  } else {
    return nullptr;
  }
}

bi::Type* bi::OptionalType::common(const WeakType& o) const {
  auto single1 = single->common(o);
  if (single1) {
    return new OptionalType(single1);
  } else {
    return nullptr;
  }
}

bi::Type* bi::OptionalType::common(const SequenceType& o) const {
  auto single1 = single->common(o);
  if (single1) {
    return new OptionalType(single1);
  } else {
    return nullptr;
  }
}

bi::Type* bi::OptionalType::common(const TupleType& o) const {
  auto single1 = single->common(o);
  if (single1) {
    return new OptionalType(single1);
  } else {
    return nullptr;
  }
}

bi::Type* bi::OptionalType::common(const UnknownType& o) const {
  auto single1 = single->common(o);
  if (single1) {
    return new OptionalType(single1);
  } else {
    return nullptr;
  }
}

bi::Type* bi::OptionalType::common(const TypeList& o) const {
  auto single1 = single->common(o);
  if (single1) {
    return new OptionalType(single1);
  } else {
    return nullptr;
  }
}
