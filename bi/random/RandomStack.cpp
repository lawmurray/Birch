/**
 * @file
 */
#include "bi/random/RandomStack.hpp"

void bi::RandomStack::pop(const int pos) {
  Expirable* top;
  while (randoms.size() > pos) {
    top = randoms.top();;
    randoms.pop();
    top->expire();
    delete top;
  }
}

bi::RandomStack randomStack;
