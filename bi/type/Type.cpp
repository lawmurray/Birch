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

bi::Type::operator bool() const {
  return true;
}

bool bi::Type::builtin() const {
  return false;
}

int bi::Type::count() const {
  return 0;
}

bool bi::Type::operator<(const Type& o) const {
  return *this <= o && *this != o;
}

bool bi::Type::operator>(const Type& o) const {
  return o <= *this && o != *this;
}

bool bi::Type::operator>=(const Type& o) const {
  return o <= *this;
}

bool bi::Type::operator!=(const Type& o) const {
  return !(*this == o);
}
