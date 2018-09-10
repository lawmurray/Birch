/**
 * @file
 */
#include "bi/type/MemberType.hpp"

#include "bi/visitor/all.hpp"

bi::MemberType::MemberType(Type* left, Type* right, Location* loc) :
    Type(loc),
    Couple<Type>(left, right) {
  //
}

bi::MemberType::~MemberType() {
  //
}

bi::Type* bi::MemberType::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Type* bi::MemberType::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::MemberType::accept(Visitor* visitor) const {
  return visitor->visit(this);
}

bool bi::MemberType::isBasic() const {
  return right->isBasic();
}

bool bi::MemberType::isClass() const {
  return right->isClass();
}

bool bi::MemberType::isPointer() const {
  return right->isPointer();
}

bool bi::MemberType::isArray() const {
  return right->isArray();
}

bool bi::MemberType::isFunction() const {
  return right->isFunction();
}

bool bi::MemberType::isFiber() const {
  return right->isFiber();
}

bool bi::MemberType::isMember() const {
  return true;
}

int bi::MemberType::depth() const {
  return right->depth();
}

bi::Basic* bi::MemberType::getBasic() const {
  return right->getBasic();
}

bi::Class* bi::MemberType::getClass() const {
  return right->getClass();
}

bi::Type* bi::MemberType::canonical() {
  return right->canonical();
}

const bi::Type* bi::MemberType::canonical() const {
  return right->canonical();
}

bi::FunctionType* bi::MemberType::resolve(Argumented* o) {
  return right->resolve(o);
}

void bi::MemberType::resolveConstructor(Argumented* o) {
  right->resolveConstructor(o);
}

bool bi::MemberType::dispatchDefinitely(const Type& o) const {
  return o.definitely(*this);
}

bool bi::MemberType::definitely(const ArrayType& o) const {
  return right->definitely(o);
}

bool bi::MemberType::definitely(const BasicType& o) const {
  return right->definitely(o);
}

bool bi::MemberType::definitely(const ClassType& o) const {
  return right->definitely(o);
}

bool bi::MemberType::definitely(const EmptyType& o) const {
  return right->definitely(o);
}

bool bi::MemberType::definitely(const FiberType& o) const {
  return right->definitely(o);
}

bool bi::MemberType::definitely(const FunctionType& o) const {
  return right->definitely(o);
}

bool bi::MemberType::definitely(const GenericType& o) const {
  return right->definitely(o);
}

bool bi::MemberType::definitely(const MemberType& o) const {
  return right->definitely(o);
}

bool bi::MemberType::definitely(const OptionalType& o) const {
  return right->definitely(o);
}

bool bi::MemberType::definitely(const PointerType& o) const {
  return right->definitely(o);
}

bool bi::MemberType::definitely(const SequenceType& o) const {
  return right->definitely(o);
}

bool bi::MemberType::definitely(const TupleType& o) const {
  return right->definitely(o);
}

bool bi::MemberType::definitely(const TypeList& o) const {
  return right->definitely(o);
}

bi::Type* bi::MemberType::dispatchCommon(const Type& o) const {
  return o.common(*this);
}

bi::Type* bi::MemberType::common(const ArrayType& o) const {
  return right->common(o);
}

bi::Type* bi::MemberType::common(const BasicType& o) const {
  return right->common(o);
}

bi::Type* bi::MemberType::common(const ClassType& o) const {
  return right->common(o);
}

bi::Type* bi::MemberType::common(const EmptyType& o) const {
  return right->common(o);
}

bi::Type* bi::MemberType::common(const FiberType& o) const {
  return right->common(o);
}

bi::Type* bi::MemberType::common(const FunctionType& o) const {
  return right->common(o);
}

bi::Type* bi::MemberType::common(const GenericType& o) const {
  return right->common(o);
}

bi::Type* bi::MemberType::common(const MemberType& o) const {
  return right->common(o);
}

bi::Type* bi::MemberType::common(const OptionalType& o) const {
  return right->common(o);
}

bi::Type* bi::MemberType::common(const PointerType& o) const {
  return right->common(o);
}

bi::Type* bi::MemberType::common(const SequenceType& o) const {
  return right->common(o);
}

bi::Type* bi::MemberType::common(const TupleType& o) const {
  return right->common(o);
}

bi::Type* bi::MemberType::common(const TypeList& o) const {
  return right->common(o);
}
