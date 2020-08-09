/**
 * @file
 */
#include "bi/common/Conditioned.hpp"

bi::Conditioned::Conditioned(Expression* cond) :
    cond(cond) {
  /* pre-condition */
  assert(cond);
}

bi::Conditioned::~Conditioned() {
  //
}
