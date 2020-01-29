/**
 * @file
 */
#include "bi/type/ClassType.hpp"

#include "bi/visitor/all.hpp"

bool bi::allowConversions = true;

bi::ClassType::ClassType(const bool weak, Name* name, Type* typeArgs,
    Location* loc, Class* target) :
    Type(loc),
    Named(name),
    TypeArgumented(typeArgs),
    Reference<Class>(target),
    weak(weak) {
  //
}

bi::ClassType::ClassType(Class* target, Location* loc) :
    Type(loc),
    Named(target->name),
    TypeArgumented(new EmptyType(loc)),
    Reference<Class>(target),
    weak(false) {
  //
}

bi::ClassType::~ClassType() {
  //
}

bi::Type* bi::ClassType::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Type* bi::ClassType::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::ClassType::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::ClassType::isClass() const {
  return true;
}

bi::Class* bi::ClassType::getClass() const {
  return target;
}

bi::Type* bi::ClassType::canonical() {
  assert(target);
  if (target->alias) {
    return target->base->canonical();
  } else {
    return this;
  }
}

const bi::Type* bi::ClassType::canonical() const {
  assert(target);
  if (target->alias) {
    return target->base->canonical();
  } else {
    return this;
  }
}

void bi::ClassType::resolveConstructor(Argumented* o) {
  assert(target);
  bi::is_convertible compare;
  if (!compare(o, target)) {
    throw ConstructorException(o, target);
  }
}

bool bi::ClassType::dispatchIsConvertible(const Type& o) const {
  return o.isConvertible(*this);
}

bool bi::ClassType::isConvertible(const GenericType& o) const {
  assert(o.target);
  return isConvertible(*o.target->type);
}

bool bi::ClassType::isConvertible(const MemberType& o) const {
  return isConvertible(*o.right);
}

bool bi::ClassType::isConvertible(const ArrayType& o) const {
  assert(target);
  return (allowConversions && target->hasConversion(&o)) ||
      target->base->isConvertible(o);
}

bool bi::ClassType::isConvertible(const BasicType& o) const {
  assert(target);
  return (allowConversions && target->hasConversion(&o)) ||
      target->base->isConvertible(o);
}

bool bi::ClassType::isConvertible(const ClassType& o) const {
  assert(target);
  auto o1 = o.canonical();
  return target == o1->getClass() || target->hasSuper(o1)
      || (allowConversions && target->hasConversion(&o));
}

bool bi::ClassType::isConvertible(const FiberType& o) const {
  assert(target);
  return (allowConversions && target->hasConversion(&o)) ||
      target->base->isConvertible(o);
}

bool bi::ClassType::isConvertible(const FunctionType& o) const {
  assert(target);
  return (allowConversions && target->hasConversion(&o)) ||
      target->base->isConvertible(o);
}

bool bi::ClassType::isConvertible(const OptionalType& o) const {
  assert(target);
  return isConvertible(*o.single) || (allowConversions &&
      target->hasConversion(&o)) || target->base->isConvertible(o);
}

bool bi::ClassType::isConvertible(const TupleType& o) const {
  assert(target);
  return (allowConversions && target->hasConversion(&o)) ||
      target->base->isConvertible(o);
}

bool bi::ClassType::dispatchIsAssignable(const Type& o) const {
  return o.isAssignable(*this);
}

bool bi::ClassType::isAssignable(const GenericType& o) const {
  assert(o.target);
  return isAssignable(*o.target->type);
}

bool bi::ClassType::isAssignable(const MemberType& o) const {
  return isAssignable(*o.right);
}

bool bi::ClassType::isAssignable(const ArrayType& o) const {
  assert(target);
  return (allowConversions && target->hasConversion(&o)) ||
      target->base->isAssignable(o);
}

bool bi::ClassType::isAssignable(const BasicType& o) const {
  assert(target);
  return (allowConversions && target->hasConversion(&o)) ||
      target->base->isAssignable(o);
}

bool bi::ClassType::isAssignable(const ClassType& o) const {
  assert(target);
  auto o1 = o.canonical();
  return target == o1->getClass() || target->hasSuper(o1)
      || (allowConversions && target->hasConversion(&o));
}

bool bi::ClassType::isAssignable(const FiberType& o) const {
  assert(target);
  return (allowConversions && target->hasConversion(&o)) ||
      target->base->isAssignable(o);
}

bool bi::ClassType::isAssignable(const FunctionType& o) const {
  assert(target);
  return (allowConversions && target->hasConversion(&o)) ||
      target->base->isAssignable(o);
}

bool bi::ClassType::isAssignable(const OptionalType& o) const {
  assert(target);
  return isAssignable(*o.single) || (allowConversions &&
      target->hasConversion(&o)) || target->base->isAssignable(o);
}

bool bi::ClassType::isAssignable(const TupleType& o) const {
  assert(target);
  return (allowConversions && target->hasConversion(&o)) ||
      target->base->isAssignable(o);
}
