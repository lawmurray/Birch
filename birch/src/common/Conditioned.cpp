/**
 * @file
 */
#include "src/common/Conditioned.hpp"

birch::Conditioned::Conditioned(Expression* cond) :
    cond(cond) {
  /* pre-condition */
  assert(cond);
}
