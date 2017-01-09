/**
 * @file
 */
#include "bi/random/Expirable.hpp"

#include <cassert>

bi::Expirable::Expirable() : expired(false) {
  //
}

void bi::Expirable::expire() {
  /* pre-condition */
  assert(!expired);

  expired = true;
}

bool bi::Expirable::isExpired() const {
  return expired;
}
