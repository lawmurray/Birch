/**
 * @file
 */
#pragma once

#include "bi/random/Expirable.hpp"

#include <stack>

namespace bi {
/**
 * Stack of random variables.
 */
class RandomStack {
public:
  /**
   * Push a random variable onto the stack and return its position.
   */
  int push(Expirable* random);

  /**
   * Pop all random variables down to and including the given position on the
   * stack.
   */
  void pop(const int pos);

private:
  /**
   * Stack of random variables.
   */
  std::stack<Expirable*> randoms;
};
}

/**
 * The singleton random stack.
 */
extern bi::RandomStack randomStack;
