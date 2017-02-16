/**
 * @file
 */
#include "bi/method/Stack.hpp"

#include <cassert>

bi::Stack::Stack() :
    logLikelihood(0.0) {
  //
}

int bi::Stack::add(random_canonical* rv, const int state) {
  if (state == -1) {
    logLikelihood += rv->backward();
    delete rv;
    return state;
  } else {
    canonicals.push(rv);
    return canonicals.size() - 1;
  }
}

bi::random_canonical* bi::Stack::get(const int state) {
  pop(state + 1);
  return canonicals.top();
}

void bi::Stack::simulate(const int state) {
  pop(state);
}

void bi::Stack::pop(const int state) {
  /* pre-condition */
  assert(state >= 0);

  while (canonicals.size() > state) {
    auto* rv = canonicals.top();
    canonicals.pop();

    rv->simulate();
    logLikelihood += rv->backward();
    delete rv;
  }
}
