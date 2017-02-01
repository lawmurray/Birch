/**
 * @file
 */
#include "bi/type/Type.hpp"

#include "bi/visitor/IsRandom.hpp"

bi::Type::Type(shared_ptr<Location> loc) :
    Located(loc),
    assignable(false) {
  //
}

bi::Type::~Type() {
  //
}

bool bi::Type::isEmpty() const {
  return false;
}

bool bi::Type::isRandom() const {
  IsRandom visitor;
  this->accept(&visitor);
  return visitor.result;
}

bool bi::Type::isBuiltin() const {
  return false;
}

bool bi::Type::isModel() const {
  return false;
}

bi::Type* bi::Type::strip() {
  return this;
}

int bi::Type::count() const {
  return 0;
}

bi::possibly bi::Type::operator<=(Type& o) {
  return o.dispatch(*this);
}


bi::possibly bi::Type::operator==(Type& o) {
  return *this <= o && o <= *this;
}

bi::possibly bi::Type::le(AssignableType& o) {
  return untrue;
}

bi::possibly bi::Type::le(BracketsType& o) {
  return untrue;
}

bi::possibly bi::Type::le(EmptyType& o) {
  return untrue;
}

bi::possibly bi::Type::le(List<Type>& o) {
  return untrue;
}

bi::possibly bi::Type::le(ModelParameter& o) {
  return untrue;
}

bi::possibly bi::Type::le(ModelReference& o) {
  return untrue;
}

bi::possibly bi::Type::le(ParenthesesType& o) {
  return untrue;
}

bi::possibly bi::Type::le(RandomType& o) {
  return untrue;
}
