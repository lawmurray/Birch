/**
 * @file
 */
#include "bi/random/RandomStack.hpp"

int bi::RandomStack::push(Expirable* random) {
  randoms.push(random);
  return randoms.size() - 1;
}

void bi::RandomStack::pop(const int pos) {
  while (randoms.size() > pos) {
    randoms.top()->expire();
    randoms.pop();
  }
}

bi::RandomStack randomStack;
