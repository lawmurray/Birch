/**
 * @file
 */
#include "bi/type/ClassType.hpp"

#include "bi/visitor/all.hpp"

bi::ClassType::ClassType(Name* name, Type* typeArgs, Location* loc,
    Class* target) :
    Type(loc),
    Named(name),
    Reference<Class>(target),
    typeArgs(typeArgs),
    original(nullptr) {
  //
}

bi::ClassType::ClassType(Class* target, Location* loc) :
    Type(loc),
    Named(target->name),
    Reference<Class>(target),
    typeArgs(new EmptyType(loc)),
    original(nullptr) {
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
  bi::definitely compare;
  if (!compare(o, target)) {
    throw ConstructorException(o, target);
  }
}

bool bi::ClassType::dispatchDefinitely(const Type& o) const {
  return o.definitely(*this);
}

bool bi::ClassType::definitely(const GenericType& o) const {
  assert(o.target);
  return definitely(*o.target->type);
}

bool bi::ClassType::definitely(const ArrayType& o) const {
  assert(target);
  return target->hasConversion(&o) || target->base->definitely(o);
}

bool bi::ClassType::definitely(const BasicType& o) const {
  assert(target);
  return target->hasConversion(&o) || target->base->definitely(o);
}

bool bi::ClassType::definitely(const ClassType& o) const {
  assert(target);
  auto o1 = o.canonical();
  return target == o1->getClass() || target->hasSuper(o1)
      || target->hasConversion(o1);
}

bool bi::ClassType::definitely(const FiberType& o) const {
  assert(target);
  return target->hasConversion(&o) || target->base->definitely(o);
}

bool bi::ClassType::definitely(const FunctionType& o) const {
  assert(target);
  return target->hasConversion(&o) || target->base->definitely(o);
}

bool bi::ClassType::definitely(const OptionalType& o) const {
  return definitely(*o.single) || target->hasConversion(&o)
      || target->base->definitely(o);
}

bool bi::ClassType::definitely(const TupleType& o) const {
  assert(target);
  return target->hasConversion(&o) || target->base->definitely(o);
}

bool bi::ClassType::definitely(const AnyType& o) const {
  return true;
}

bi::Type* bi::ClassType::dispatchCommon(const Type& o) const {
  return o.common(*this);
}

bi::Type* bi::ClassType::common(const GenericType& o) const {
  assert(o.target);
  return common(*o.target->type);
}

bi::Type* bi::ClassType::common(const ArrayType& o) const {
  assert(target);
  if (definitely(o)) {
    return o.common(o);
  } else {
    return nullptr;
  }
}

bi::Type* bi::ClassType::common(const BasicType& o) const {
  assert(target);
  if (definitely(o)) {
    return o.common(o);
  } else {
    return nullptr;
  }
}

bi::Type* bi::ClassType::common(const ClassType& o) const {
  assert(target);
  assert(o.target);
  if (target == o.target) {
    return new ClassType(target);
  } else if (target->hasSuper(&o)) {
    return new ClassType(o.target);
  } else if (o.target->hasSuper(this)) {
    return new ClassType(target);
  } else if (definitely(o)) {
    return o.common(o);
  } else {
    return nullptr;
  }
}

bi::Type* bi::ClassType::common(const FiberType& o) const {
  assert(target);
  if (definitely(o)) {
    return o.common(o);
  } else {
    return nullptr;
  }
}

bi::Type* bi::ClassType::common(const FunctionType& o) const {
  assert(target);
  if (definitely(o)) {
    return o.common(o);
  } else {
    return nullptr;
  }
}

bi::Type* bi::ClassType::common(const OptionalType& o) const {
  auto single1 = common(*o.single);
  if (single1) {
    return new OptionalType(single1);
  } else if (definitely(o)) {
    return o.common(o);
  } else {
    return nullptr;
  }
}

bi::Type* bi::ClassType::common(const TupleType& o) const {
  assert(target);
  if (definitely(o)) {
    return o.common(o);
  } else {
    return nullptr;
  }
}

bi::Type* bi::ClassType::common(const AnyType& o) const {
  return new AnyType();
}
