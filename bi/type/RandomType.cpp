/**
 * @file
 */
#include "bi/type/RandomType.hpp"

#include "bi/type/ModelParameter.hpp"
#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::RandomType::RandomType(Type* left, Type* right, shared_ptr<Location> loc) :
    Type(loc),
    TypeBinary(left, right) {
  //
}

bi::RandomType::~RandomType() {
  //
}

bi::Type* bi::RandomType::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Type* bi::RandomType::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::RandomType::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bi::possibly bi::RandomType::dispatch(Type& o) {
  return o.le(*this);
}

bi::possibly bi::RandomType::le(EmptyType& o) {
  return possibly(!o.assignable || assignable);
}

bi::possibly bi::RandomType::le(List<Type>& o) {
  return (*left <= o || (possible && *right <= o))
      && (!o.assignable || assignable);
}

bi::possibly bi::RandomType::le(ModelReference& o) {
  return (*left <= o || (possible && *right <= o))
      && (!o.assignable || assignable);
}

bi::possibly bi::RandomType::le(ModelParameter& o) {
  return (*left <= o || (possible && *right <= o))
      && (!o.assignable || assignable);
}

bi::possibly bi::RandomType::le(RandomType& o) {
  return possible && *left <= *o.left && *right <= *o.right
      && (!o.assignable || assignable);
}
