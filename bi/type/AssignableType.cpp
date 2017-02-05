/**
 * @file
 */
#include "bi/type/AssignableType.hpp"

#include "bi/visitor/all.hpp"

bi::AssignableType::AssignableType(Type* single, shared_ptr<Location> loc) :
    Type(loc),
    TypeUnary(single) {
  //
}

bi::AssignableType::~AssignableType() {
  //
}

bool bi::AssignableType::isBuiltin() const {
  return single->isBuiltin();
}

bool bi::AssignableType::isModel() const {
  return single->isModel();
}

bi::Type* bi::AssignableType::strip() {
  return single->strip();
}

bi::Type* bi::AssignableType::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Type* bi::AssignableType::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::AssignableType::accept(Visitor* visitor) const {
  return visitor->visit(this);
}

bool bi::AssignableType::dispatchDefinitely(Type& o) {
  return o.definitely(*this) || single->dispatchDefinitely(o);
}

bool bi::AssignableType::definitely(AssignableType& o) {
  return single->definitely(*o.single);
}

bool bi::AssignableType::definitely(BracketsType& o) {
  return single->definitely(o);
}

bool bi::AssignableType::definitely(EmptyType& o) {
  return single->definitely(o);
}

bool bi::AssignableType::definitely(List<Type>& o) {
  return single->definitely(o);
}

bool bi::AssignableType::definitely(ModelParameter& o) {
  return single->definitely(o);
}

bool bi::AssignableType::definitely(ModelReference& o) {
  return single->definitely(o);
}

bool bi::AssignableType::definitely(ParenthesesType& o) {
  return single->definitely(*o.single);
}

bool bi::AssignableType::dispatchPossibly(Type& o) {
  return o.possibly(*this) || single->dispatchPossibly(o);
}

bool bi::AssignableType::possibly(AssignableType& o) {
  return single->possibly(*o.single);
}

bool bi::AssignableType::possibly(BracketsType& o) {
  return single->possibly(o);
}

bool bi::AssignableType::possibly(EmptyType& o) {
  return single->possibly(o);
}

bool bi::AssignableType::possibly(List<Type>& o) {
  return single->possibly(o);
}

bool bi::AssignableType::possibly(ModelParameter& o) {
  return single->possibly(o);
}

bool bi::AssignableType::possibly(ModelReference& o) {
  return single->possibly(o);
}

bool bi::AssignableType::possibly(ParenthesesType& o) {
  return single->possibly(*o.single);
}
