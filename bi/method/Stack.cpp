/**
 * @file
 */
#include "bi/method/Stack.hpp"

#include <cassert>

bi::Stack::Stack() :
    logWeight(0.0) {
  //
}

int bi::Stack::add(DelayInterface* o) {
  if (o->getState() == MISSING) {
    /* missing value, put on stack for later */
    stack.push(o);
    return stack.size() - 1;
  } else {
    /* known value, observe now */
    logWeight += o->observe();
    return -1;
  }
}

bi::DelayInterface* bi::Stack::get(const int id) {
  /* pre-condition */
  assert(0 <= id && id < stack.size());

  /* ensure variate is on top of the stack */
  pop(id + 1);

  return stack.top();
}

void bi::Stack::sample(const int id) {
  /* pre-condition */
  assert(0 <= id && id < stack.size());

  /* ensure variate is on top of the stack */
  pop(id + 1);

  /* sample the variate */
  DelayInterface* o = stack.top();
  assert(o->getState() == MISSING);
  o->sample();
  o->setId(-1);
  o->setState(SIMULATED);
  logWeight += o->observe();
  stack.pop();
}

void bi::Stack::observe(const int id) {
  /* pre-condition */
  assert(0 <= id && id < stack.size());

  /* ensure variate is on top of the stack */
  pop(id + 1);

  /* observe the variate */
  DelayInterface* o = stack.top();
  assert(o->getState() == MISSING);
  o->setId(-1);
  o->setState(ASSIGNED);
  logWeight += o->observe();
  stack.pop();
}

void bi::Stack::pop(const int id) {
  /* pre-condition */
  assert(id >= 0);

  while (stack.size() > id) {
    DelayInterface* o = stack.top();
    assert(o->getState() == MISSING);
    o->sample();
    o->setId(-1);
    o->setState(SIMULATED);
    logWeight += o->observe();
    stack.pop();
  }
}
