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

const bi::Type* bi::PointerType::unwrap() const {
  return single;
}

bi::Type* bi::PointerType::unwrap() {
  return single;
}

bool bi::PointerType::dispatchDefinitely(const Type& o) const {
  return o.definitely(*this);
}

bool bi::PointerType::definitely(const GenericType& o) const {
  assert(o.target);
  return definitely(*o.target->type);
}

bool bi::PointerType::definitely(const OptionalType& o) const {
  return single->definitely(*o.single);
}

bool bi::PointerType::definitely(const PointerType& o) const {
  return single->definitely(*o.single) && (!weak || o.weak)
      && (!read || o.read);
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

bi::Type* bi::PointerType::common(const OptionalType& o) const {
  auto single1 = single->common(*o.single);
  if (single1) {
    return new OptionalType(single1);
  } else {
    return nullptr;
  }
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
