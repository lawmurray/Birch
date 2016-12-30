/**
 * @file
 */
#include "bi/common/Braced.hpp"

bi::Braced::Braced(Expression* braces) :
    braces(braces) {
  /* pre-condition */
  assert(braces);
}

bi::Braced::~Braced() {
  //
}
