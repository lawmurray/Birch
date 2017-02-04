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

bi::possibly bi::AssignableType::dispatch(Type& o) {
  return o.le(*this) || single->dispatch(o);
}

bi::possibly bi::AssignableType::le(AssignableType& o) {
  return *single <= *o.single;
}

bi::possibly bi::AssignableType::le(BracketsType& o) {
  return *single <= o;
}

bi::possibly bi::AssignableType::le(EmptyType& o) {
  return *single <= o;
}

bi::possibly bi::AssignableType::le(List<Type>& o) {
  return *single <= o;
}

bi::possibly bi::AssignableType::le(ModelParameter& o) {
  return *single <= o;
}

bi::possibly bi::AssignableType::le(ModelReference& o) {
  return *single <= o;
}

bi::possibly bi::AssignableType::le(ParenthesesType& o) {
  return *single <= *o.single;
}

bi::possibly bi::AssignableType::le(RandomType& o) {
  return *single <= o;
}
