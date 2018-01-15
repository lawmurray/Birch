/**
 * @file
 */
#include "bi/type/PointerType.hpp"

#include "bi/visitor/all.hpp"

bi::PointerType::PointerType(const bool weak, Type* single, const bool read,
    Location* loc) :
    Type(loc),
    Single<Type>(single),
    weak(weak),
    read(read) {
  //
}

bi::PointerType::~PointerType() {
  //
}

bi::Type* bi::PointerType::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Type* bi::PointerType::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::PointerType::accept(Visitor* visitor) const {
  return visitor->visit(this);
}

bool bi::PointerType::isPointer() const {
  return true;
}

bool bi::PointerType::isWeak() const {
  return weak;
}

const bi::Type* bi::PointerType::unwrap() const {
  return single;
}

bi::Type* bi::PointerType::unwrap() {
  return single;
}

void bi::PointerType::resolveConstructor(Argumented* o) {
  if (!weak) {
    single->resolveConstructor(o);
  } else if (!o->args->isEmpty()) {
    throw ConstructorException(o);
  }
}

bool bi::PointerType::dispatchDefinitely(const Type& o) const {
  return o.definitely(*this);
}

bool bi::PointerType::definitely(const GenericType& o) const {
  assert(o.target);
  return definitely(*o.target->type);
}

bool bi::PointerType::definitely(const ArrayType& o) const {
  return !weak && single->definitely(o);
}

bool bi::PointerType::definitely(const BasicType& o) const {
  return !weak && single->definitely(o);
}

bool bi::PointerType::definitely(const ClassType& o) const {
  return !weak && single->definitely(o);
}

bool bi::PointerType::definitely(const FiberType& o) const {
  return !weak && single->definitely(o);
}

bool bi::PointerType::definitely(const FunctionType& o) const {
  return !weak && single->definitely(o);
}

bool bi::PointerType::definitely(const OptionalType& o) const {
  if (o.single->isPointer()) {
    /* a weak pointer can be assigned to an optional of a shared or weak
     * pointer */
    return single->definitely(*o.single->unwrap());
  } else {
    return definitely(*o.single);
  }
}

bool bi::PointerType::definitely(const TupleType& o) const {
  return !weak && single->definitely(o);
}

bool bi::PointerType::definitely(const PointerType& o) const {
  return (!weak || o.weak) && (!read || o.read)
      && single->definitely(*o.single);
}

bool bi::PointerType::definitely(const AnyType& o) const {
  return true;
}

bi::Type* bi::PointerType::dispatchCommon(const Type& o) const {
  return o.common(*this);
}

bi::Type* bi::PointerType::common(const GenericType& o) const {
  assert(o.target);
  return common(*o.target->type);
}

bi::Type* bi::PointerType::common(const ArrayType& o) const {
  return weak ? nullptr : single->common(o);
}

bi::Type* bi::PointerType::common(const BasicType& o) const {
  return weak ? nullptr : single->common(o);
}

bi::Type* bi::PointerType::common(const ClassType& o) const {
  return weak ? nullptr : single->common(o);
}

bi::Type* bi::PointerType::common(const FiberType& o) const {
  return weak ? nullptr : single->common(o);
}

bi::Type* bi::PointerType::common(const FunctionType& o) const {
  return weak ? nullptr : single->common(o);
}

bi::Type* bi::PointerType::common(const OptionalType& o) const {
  if (o.single->isPointer()) {
    /* the least common type of any pointer and an optional of any pointer is
     * the latter, as e.g. for A& and A?, an A& can be assigned to an A? but
     * not vice-versa, and both can be assigned to a A&?, so A? is the least
     * common type */
    auto single1 = dynamic_cast<PointerType*>(common(*o.single));
    if (single1) {
      single1->weak = false;
      return new OptionalType(single1);
    } else {
      return nullptr;
    }
  } else {
    auto single1 = common(*o.single);
    if (single1) {
      return new OptionalType(single1);
    } else {
      return nullptr;
    }
  }
}

bi::Type* bi::PointerType::common(const TupleType& o) const {
  return weak ? nullptr : single->common(o);
}

bi::Type* bi::PointerType::common(const PointerType& o) const {
  auto single1 = single->common(*o.single);
  if (single1) {
    return new PointerType(weak || o.weak, single1, read || o.read);
  } else {
    return nullptr;
  }
}

bi::Type* bi::PointerType::common(const AnyType& o) const {
  return new AnyType();
}
