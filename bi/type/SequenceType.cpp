/**
 * @file
 */
#include "bi/type/SequenceType.hpp"

#include "bi/visitor/all.hpp"

bi::SequenceType::SequenceType(Type* single, Location* loc) :
    Type(loc),
    Single<Type>(single) {
  //
}

bi::SequenceType::~SequenceType() {
  //
}

int bi::SequenceType::depth() const {
  return 1 + single->depth();
}

bool bi::SequenceType::isValue() const {
  return single->isValue();
}

bool bi::SequenceType::isSequence() const {
  return true;
}

bi::Type* bi::SequenceType::element() {
  return single->element();
}

const bi::Type* bi::SequenceType::element() const {
  return single->element();
}

bi::Type* bi::SequenceType::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Type* bi::SequenceType::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::SequenceType::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::SequenceType::dispatchIsConvertible(const Type& o) const {
  return o.isConvertible(*this);
}

bool bi::SequenceType::isConvertible(const ArrayType& o) const {
  return single->element()->isConvertible(*o.single->element())
      && depth() == o.depth();
}

bool bi::SequenceType::isConvertible(const GenericType& o) const {
  assert(o.target);
  return isConvertible(*o.target->type);
}

bool bi::SequenceType::isConvertible(const MemberType& o) const {
  return isConvertible(*o.right);
}

bool bi::SequenceType::isConvertible(const OptionalType& o) const {
  return isConvertible(*o.single);
}

bool bi::SequenceType::isConvertible(const SequenceType& o) const {
  return single->isConvertible(*o.single);
}

bool bi::SequenceType::dispatchIsAssignable(const Type& o) const {
  return o.isAssignable(*this);
}

bool bi::SequenceType::isAssignable(const ArrayType& o) const {
  return single->element()->isAssignable(*o.single->element())
      && depth() == o.depth();
}

bool bi::SequenceType::isAssignable(const ClassType& o) const {
  return o.getClass()->hasAssignment(this);
}

bool bi::SequenceType::isAssignable(const GenericType& o) const {
  assert(o.target);
  return isAssignable(*o.target->type);
}

bool bi::SequenceType::isAssignable(const MemberType& o) const {
  return isAssignable(*o.right);
}

bool bi::SequenceType::isAssignable(const OptionalType& o) const {
  return isAssignable(*o.single);
}

bool bi::SequenceType::isAssignable(const SequenceType& o) const {
  return single->isAssignable(*o.single);
}

bi::Type* bi::SequenceType::dispatchCommon(const Type& o) const {
  return o.common(*this);
}

bi::Type* bi::SequenceType::common(const ArrayType& o) const {
  auto single1 = single->element()->common(*o.single->element());
  if (single1 && depth() == o.depth()) {
    return new ArrayType(single1, depth());
  } else {
    return nullptr;
  }
}

bi::Type* bi::SequenceType::common(const GenericType& o) const {
  assert(o.target);
  return common(*o.target->type);
}

bi::Type* bi::SequenceType::common(const MemberType& o) const {
  return common(*o.right);
}

bi::Type* bi::SequenceType::common(const OptionalType& o) const {
  auto single1 = common(*o.single);
  if (single1) {
    return new OptionalType(single1);
  } else {
    return nullptr;
  }
}

bi::Type* bi::SequenceType::common(const SequenceType& o) const {
  auto single1 = single->common(*o.single);
  if (single1) {
    return new SequenceType(single1);
  } else {
    return nullptr;
  }
}
