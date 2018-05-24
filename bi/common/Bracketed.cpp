/**
 * @file
 */
#include "bi/common/Bracketed.hpp"

bi::Bracketed::Bracketed(Expression* brackets) :
    brackets(brackets) {
  /* pre-condition */
  assert(brackets);
}

bi::Bracketed::~Bracketed() {
  //
}
