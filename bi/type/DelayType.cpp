/**
 * @file
 */
#include "bi/type/DelayType.hpp"

#include "bi/type/ModelParameter.hpp"
#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::DelayType::DelayType(Type* left, Type* right, shared_ptr<Location> loc,
    const bool assignable) :
    Type(loc, assignable),
    TypeBinary(left, right) {
  //
}

bi::DelayType::~DelayType() {
  //
}

bi::Type* bi::DelayType::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Type* bi::DelayType::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::DelayType::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::DelayType::isDelay() const {
  return true;
}

bool bi::DelayType::dispatchDefinitely(const Type& o) const {
  return o.definitely(*this);
}

bool bi::DelayType::definitely(const DelayType& o) const {
  return left->definitely(*o.left) && right->definitely(*o.right)
      && (!o.assignable || assignable);
}

bool bi::DelayType::definitely(const LambdaType& o) const {
  return left->definitely(o) && (!o.assignable || assignable);
}

bool bi::DelayType::definitely(const List<Type>& o) const {
  return left->definitely(o) && (!o.assignable || assignable);
}

bool bi::DelayType::definitely(const ModelReference& o) const {
  return left->definitely(o) && (!o.assignable || assignable);
}

bool bi::DelayType::definitely(const ModelParameter& o) const {
  return left->definitely(o) && (!o.assignable || assignable);
}

bool bi::DelayType::definitely(const ParenthesesType& o) const {
  return definitely(*o.single) && (!o.assignable || assignable);
}

bool bi::DelayType::dispatchPossibly(const Type& o) const {
  return o.possibly(*this);
}

bool bi::DelayType::possibly(const DelayType& o) const {
  return left->possibly(*o.left) && right->possibly(*o.right)
      && (!o.assignable || assignable);
}

bool bi::DelayType::possibly(const LambdaType& o) const {
  return (left->possibly(o) || right->possibly(o))
      && (!o.assignable || assignable);
}

bool bi::DelayType::possibly(const List<Type>& o) const {
  return (left->possibly(o) || right->possibly(o))
      && (!o.assignable || assignable);
}

bool bi::DelayType::possibly(const ModelReference& o) const {
  return (left->possibly(o) || right->possibly(o))
      && (!o.assignable || assignable);
}

bool bi::DelayType::possibly(const ModelParameter& o) const {
  return (left->possibly(o) || right->possibly(o))
      && (!o.assignable || assignable);
}

bool bi::DelayType::possibly(const ParenthesesType& o) const {
  return possibly(*o.single) && (!o.assignable || assignable);
}
