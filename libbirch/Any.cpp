/**
 * @file
 */
#include "libbirch/Any.hpp"

bi::Any::Any() {
  //
}

bi::Any::Any(const Any& o) :
    ptr() {
  //
}

bi::Any::~Any() {
  //
}

bi::Any* bi::Any::clone() {
  return new Any(*this);
}
