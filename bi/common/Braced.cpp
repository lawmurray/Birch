/**
 * @file
 */
#include "bi/common/Braced.hpp"

bi::Braced::Braced(Statement* braces) :
    braces(braces) {
  /* pre-condition */
  assert(braces);
}

bi::Braced::~Braced() {
  //
}
