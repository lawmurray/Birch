/**
 * @file
 */
#include "bi/type/RandomType.hpp"

#include "bi/type/ModelParameter.hpp"
#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::RandomType::RandomType(Type* left, Type* right, shared_ptr<Location> loc,
    const bool assignable) :
    Type(loc, assignable),
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

bool bi::RandomType::isRandom() const {
  return true;
}

bool bi::RandomType::dispatchDefinitely(const Type& o) const {
  return o.definitely(*this);
}

bool bi::RandomType::definitely(const EmptyType& o) const {
  return !o.assignable || assignable;
}

bool bi::RandomType::definitely(const LambdaType& o) const {
  return left->definitely(o) && (!o.assignable || assignable);
}

bool bi::RandomType::definitely(const List<Type>& o) const {
  return left->definitely(o) && (!o.assignable || assignable);
}

bool bi::RandomType::definitely(const ModelReference& o) const {
  return left->definitely(o) && (!o.assignable || assignable);
}

bool bi::RandomType::definitely(const ModelParameter& o) const {
  return left->definitely(o) && (!o.assignable || assignable);
}

bool bi::RandomType::definitely(const RandomType& o) const {
  return left->definitely(*o.left) && right->definitely(*o.right)
      && (!o.assignable || assignable);
}

bool bi::RandomType::dispatchPossibly(const Type& o) const {
  return o.possibly(*this);
}

bool bi::RandomType::possibly(const EmptyType& o) const {
  return !o.assignable || assignable;
}

bool bi::RandomType::possibly(const LambdaType& o) const {
  return (left->possibly(o) || right->possibly(o))
      && (!o.assignable || assignable);
}

bool bi::RandomType::possibly(const List<Type>& o) const {
  return (left->possibly(o) || right->possibly(o))
      && (!o.assignable || assignable);
}

bool bi::RandomType::possibly(const ModelReference& o) const {
  return (left->possibly(o) || right->possibly(o))
      && (!o.assignable || assignable);
}

bool bi::RandomType::possibly(const ModelParameter& o) const {
  return (left->possibly(o) || right->possibly(o))
      && (!o.assignable || assignable);
}

bool bi::RandomType::possibly(const RandomType& o) const {
  return left->possibly(*o.left) && right->possibly(*o.right)
      && (!o.assignable || assignable);
}
