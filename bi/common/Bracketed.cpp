/**
 * @file
 */
#include "bi/common/Bracketed.hpp"

#include <cassert>

bi::Bracketed::Bracketed(Expression* brackets) :
    brackets(brackets) {
  /* pre-condition */
  assert(brackets);
}

bi::Bracketed::~Bracketed() {
  //
}
