/**
 * @file
 */
#include "src/common/Bracketed.hpp"

birch::Bracketed::Bracketed(Expression* brackets) :
    brackets(brackets) {
  /* pre-condition */
  assert(brackets);
}
