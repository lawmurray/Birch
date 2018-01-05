/**
 * @file
 */
#include "libbirch/Any.hpp"

bi::Any::~Any() {
  //
}

bi::Any* bi::Any::clone() {
  return new Any(*this);
}
