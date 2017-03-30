/**
 * @file
 */
#include "bi/type/Type.hpp"

#include "bi/common/Iterator.hpp"

#include <cassert>

bi::Type::Type(shared_ptr<Location> loc, const bool assignable) :
    Located(loc),
    assignable(assignable) {
  //
}

bi::Type::~Type() {
  //
}

bool bi::Type::isEmpty() const {
  return false;
}

bool bi::Type::isBuiltin() const {
  return false;
}

bool bi::Type::isModel() const {
  return false;
}

bool bi::Type::isArray() const {
  return false;
}

bool bi::Type::isRandom() const {
  return false;
}

bool bi::Type::isLambda() const {
  return false;
}

bool bi::Type::isVariant() const {
  return false;
}

bi::Type* bi::Type::strip() {
  return this;
}

int bi::Type::count() const {
  return 0;
}

bi::Iterator<bi::Type> bi::Type::begin() const {
  return bi::Iterator<Type>(this);
}

bi::Iterator<bi::Type> bi::Type::end() const {
  return bi::Iterator<Type>(nullptr);
}

bool bi::Type::definitely(const Type& o) const {
  return o.dispatchDefinitely(*this);
}

bool bi::Type::definitely(const AssignableType& o) const {
  return false;
}

bool bi::Type::definitely(const BracketsType& o) const {
  return false;
}

bool bi::Type::definitely(const EmptyType& o) const {
  return false;
}

bool bi::Type::definitely(const LambdaType& o) const {
  return false;
}

bool bi::Type::definitely(const List<Type>& o) const {
  return false;
}

bool bi::Type::definitely(const ModelParameter& o) const {
  return false;
}

bool bi::Type::definitely(const ModelReference& o) const {
  return false;
}

bool bi::Type::definitely(const ParenthesesType& o) const {
  return false;
}

bool bi::Type::definitely(const RandomType& o) const {
  return false;
}

bool bi::Type::definitely(const VariantType& o) const {
  return false;
}

bool bi::Type::possibly(const Type& o) const {
  return o.dispatchPossibly(*this);
}

bool bi::Type::possibly(const AssignableType& o) const {
  return false;
}

bool bi::Type::possibly(const BracketsType& o) const {
  return false;
}

bool bi::Type::possibly(const EmptyType& o) const {
  return false;
}

bool bi::Type::possibly(const LambdaType& o) const {
  return false;
}

bool bi::Type::possibly(const List<Type>& o) const {
  return false;
}

bool bi::Type::possibly(const ModelParameter& o) const {
  return false;
}

bool bi::Type::possibly(const ModelReference& o) const {
  return false;
}

bool bi::Type::possibly(const ParenthesesType& o) const {
  return false;
}

bool bi::Type::possibly(const RandomType& o) const {
  return false;
}

bool bi::Type::possibly(const VariantType& o) const {
  return false;
}

bool bi::Type::equals(Type& o) {
  return definitely(o) && o.definitely(*this);
}
