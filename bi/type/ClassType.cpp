/**
 * @file
 */
#include "bi/type/ClassType.hpp"

#include "bi/visitor/all.hpp"

bi::ClassType::ClassType(Name* name, Location* loc, const bool assignable,
    Class* target) :
    Type(loc, assignable),
    Named(name),
    Reference<Class>(target) {
  //
}

bi::ClassType::ClassType(Class* target, Location* loc, const bool assignable) :
    Type(loc, assignable),
    Named(target->name),
    Reference<Class>(target) {
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

void bi::ClassType::resolveConstructor(Type* args) {
  assert(target);
  if (!args->definitely(*target->parens->type)) {
    throw ConstructorException(args, target);
  }
}

bool bi::ClassType::dispatchDefinitely(const Type& o) const {
  return o.definitely(*this);
}

bool bi::ClassType::definitely(const AliasType& o) const {
  assert(o.target);
  return definitely(*o.target->base);
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
  assert(o.target);
  return target == o.target || target->hasConversion(&o)
      || target->hasSuper(&o) || target->base->definitely(o);
}

bool bi::ClassType::definitely(const FiberType& o) const {
  assert(target);
  return target->hasConversion(&o) || target->base->definitely(o);
}

bool bi::ClassType::definitely(const FunctionType& o) const {
  assert(target);
  return target->hasConversion(&o) || target->base->definitely(o);
}

bool bi::ClassType::definitely(const ListType& o) const {
  assert(target);
  return target->hasConversion(&o) || target->base->definitely(o);
}

bool bi::ClassType::definitely(const OptionalType& o) const {
  return definitely(*o.single);
}

bool bi::ClassType::definitely(const ParenthesesType& o) const {
  return definitely(*o.single);
}

bool bi::ClassType::dispatchPossibly(const Type& o) const {
  return o.possibly(*this);
}

bool bi::ClassType::possibly(const AliasType& o) const {
  assert(o.target);
  return possibly(*o.target->base);
}

bool bi::ClassType::possibly(const ClassType& o) const {
  assert(o.target);
  return o.target->hasSuper(this);
}

bool bi::ClassType::possibly(const OptionalType& o) const {
  return possibly(*o.single);
}

bool bi::ClassType::possibly(const ParenthesesType& o) const {
  return possibly(*o.single);
}
