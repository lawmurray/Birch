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

  int add(RandomInterface* rv);
  RandomInterface* get(const int id);
  void simulate(const int id);

private:
  /**
   * Pop all random variables down to and including the given position on the
   * stack.
   */
  void pop(const int state);

  /**
   * Canonical representations of random variables.
   */
  std::stack<RandomInterface*> rvs;

  /**
   * Cumulative log-likelihood.
   */
  double logLikelihood;
};
}
