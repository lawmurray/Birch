/**
 * @file
 */
#include "bi/type/Type.hpp"

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

bool bi::Type::builtin() const {
  return false;
}

int bi::Type::count() const {
  return 0;
}

bool bi::Type::operator<=(Type& o) {
  return o.dispatch(*this);
}


bool bi::Type::operator==(Type& o) {
  return *this <= o && o <= *this;
}

bool bi::Type::le(EmptyType& o) {
  return false;
}

bool bi::Type::le(List<Type>& o) {
  return false;
}

bool bi::Type::le(ModelParameter& o) {
  return false;
}

bool bi::Type::le(ModelReference& o) {
  return false;
}

bool bi::Type::le(ParenthesesType& o) {
  return false;
}

bool bi::Type::le(RandomType& o) {
  return false;
}
