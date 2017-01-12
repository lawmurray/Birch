/**
 * @file
 */
#include "bi/type/ParenthesesType.hpp"

#include "bi/type/ModelParameter.hpp"
#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::ParenthesesType::ParenthesesType(Type* single, shared_ptr<Location> loc) :
    Type(loc),
    TypeUnary(single) {
  //
}

bi::ParenthesesType::~ParenthesesType() {
  //
}

bi::Type* bi::ParenthesesType::strip() {
  return single->strip();
}

bi::Type* bi::ParenthesesType::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Type* bi::ParenthesesType::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::ParenthesesType::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::ParenthesesType::dispatch(Type& o) {
  return o.le(*this) || single->dispatch(o);
}

bool bi::ParenthesesType::le(EmptyType& o) {
  return *single <= o;
}

bool bi::ParenthesesType::le(List<Type>& o) {
  return *single <= o;
}

bool bi::ParenthesesType::le(ModelParameter& o) {
  return *single <= o;
}

bool bi::ParenthesesType::le(ModelReference& o) {
  return *single <= o;
}

bool bi::ParenthesesType::le(ParenthesesType& o) {
  return *single <= *o.single;
}

bool bi::ParenthesesType::le(RandomType& o) {
  return *single <= o;
}
