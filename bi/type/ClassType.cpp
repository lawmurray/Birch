/**
 * @file
 */
#include "bi/type/ClassType.hpp"

#include "bi/visitor/all.hpp"

bi::ClassType::ClassType(shared_ptr<Name> name, shared_ptr<Location> loc,
    const bool assignable, Class* target) :
    Type(loc, assignable),
    Named(name),
    Reference<Class>(target) {
  //
}

bi::ClassType::ClassType(Class* target) :
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

bool bi::ClassType::dispatchDefinitely(const Type& o) const {
  return o.definitely(*this);
}

bool bi::ClassType::definitely(const AliasType& o) const {
  return definitely(*o.target->base);
}

bool bi::ClassType::definitely(const ArrayType& o) const {
  return target->hasConversion(&o);
}

bool bi::ClassType::definitely(const BasicType& o) const {
  return target->hasConversion(&o);
}

bool bi::ClassType::definitely(const ClassType& o) const {
  return target == o.target || target->hasSuper(&o) || target->hasConversion(&o);
}

bool bi::ClassType::definitely(const FiberType& o) const {
  return target->hasConversion(&o);
}

bool bi::ClassType::definitely(const FunctionType& o) const {
  return target->hasConversion(&o);
}

bool bi::ClassType::definitely(const List<Type>& o) const {
  return target->hasConversion(&o);
}

bool bi::ClassType::definitely(const Class& o) const {
  return true;
}

bool bi::ClassType::definitely(const ParenthesesType& o) const {
  return definitely(*o.single);
}

bool bi::ClassType::dispatchPossibly(const Type& o) const {
  return o.possibly(*this);
}

bool bi::ClassType::possibly(const AliasType& o) const {
  return possibly(*o.target->base);
}

bool bi::ClassType::possibly(const ClassType& o) const {
  return o.target->hasSuper(this);
}

bool bi::ClassType::possibly(const ParenthesesType& o) const {
  return possibly(*o.single);
}
