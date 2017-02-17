/**
 * @file
 */
#include "bi/method/Stack.hpp"

#include <cassert>

bi::Stack::Stack() :
    logLikelihood(0.0) {
  //
}

int bi::Stack::add(RandomInterface* rv) {
  if (rv->getState() == MISSING) {
    rvs.push(rv);
    return rvs.size() - 1;
  } else {
    logLikelihood += rv->backward();
    delete rv;
    return -1;
  }
}

bi::RandomInterface* bi::Stack::get(const int id) {
  /* pre-condition */
  assert(0 <= id && id < rvs.size());

  pop(id + 1);
  return rvs.top();
}

void bi::Stack::simulate(const int id) {
  /* pre-condition */
  assert(0 <= id && id < rvs.size());

  pop(id);
}

void bi::Stack::pop(const int id) {
  /* pre-condition */
  assert(id >= 0);

  while (rvs.size() > id) {
    auto* rv = rvs.top();
    rvs.pop();

    assert(rv->getState() == MISSING);
    rv->simulate();
    rv->setId(-1);
    rv->setState(SIMULATED);
    logLikelihood += rv->backward();

    delete rv;
  }
}
