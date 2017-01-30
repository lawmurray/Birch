/**
 * @file
 */
#pragma once

#include <stack>
#include <functional>

namespace bi {
/**
 * Stack of random variables.
 */
class RandomStack {
public:
  /**
   * Lambda type.
   */
  typedef std::function<void()> lambda_type;

  /**
   * Push a random variable onto the stack and return its position.
   */
  int push(const lambda_type& pull, const lambda_type& push);

  /**
   * Pop all random variables down to and including the given position on the
   * stack.
   */
  void pop(const int pos);

private:
  /**
   * Stack of pull functions.
   */
  std::stack<lambda_type> pulls;

  /**
   * Stack of push functions.
   */
  std::stack<lambda_type> pushes;
};
}

/**
 * The singleton random stack.
 */
extern bi::RandomStack randomStack;
