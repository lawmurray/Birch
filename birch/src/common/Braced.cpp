/**
 * @file
 */
#include "src/common/Braced.hpp"

birch::Braced::Braced(Statement* braces) :
    braces(braces) {
  /* pre-condition */
  assert(braces);
}
