/**
 * @file
 */
#pragma once

#include "bi/method/Method.hpp"

#include <stack>

namespace bi {
/**
 * Simple stack-based method for analytical inference where possible,
 * otherwise importance sampling from the prior.
 */
class Stack : public Method {
public:
  /**
   * Constructor.
   */
  Stack();

  int add(random_canonical* rv, const int state);
  random_canonical* get(const int state);
  void simulate(const int state);

private:
  /**
   * Pop all random variables down to and including the given position on the
   * stack.
   */
  void pop(const int state);

  /**
   * Canonical representations of random variables.
   */
  std::stack<random_canonical*> canonicals;

  /**
   * Cumulative log-likelihood.
   */
  double logLikelihood;
};
}
