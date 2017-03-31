/**
 * @file
 */
#pragma once

#include "bi/method/Method.hpp"

#include <stack>

namespace bi {
/**
 * Simple stack-based method for delayed samling.
 */
class Stack : public Method {
public:
  /**
   * Constructor.
   */
  Stack();

  int add(DelayInterface* rv);
  DelayInterface* get(const int id);
  void simulate(const int id);

private:
  /**
   * Pop all delay variates down to and including the given position on the
   * stack.
   */
  void pop(const int state);

  /**
   * Canonical representations of delay variates.
   */
  std::stack<DelayInterface*> rvs;

  /**
   * Cumulative log-likelihood.
   */
  double logLikelihood;
};
}
