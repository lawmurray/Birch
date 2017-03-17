/**
 * @file
 */
#pragma once

#include <list>

namespace bi {
class FuncReference;
class FuncParameter;
class Dispatcher;
class Expression;
class VarParameter;

/**
 * Captures the arguments for a function call.
 */
class ArgumentCapturer {
public:
  /**
   * Constructor.
   *
   * @param ref Function reference.
   * @param param Function parameter.
   */
  ArgumentCapturer(const FuncReference* ref, const FuncParameter* param);

  /**
   * Constructor.
   *
   * @param ref Function reference.
   * @param param Function parameter.
   */
  ArgumentCapturer(const FuncReference* ref, const Dispatcher* param);

  /**
   * Constructor.
   *
   * @param parens1 First parentheses.
   * @param parens2 Second parentheses.
   */
  ArgumentCapturer(const Expression* parens1, const Expression* parens2);

  /*
   * Iterators over results.
   */
  auto begin() {
    return gathered.begin();
  }
  auto end() {
    return gathered.end();
  }
  auto begin() const {
    return gathered.begin();
  }
  auto end() const {
    return gathered.end();
  }
  auto rbegin() {
    return gathered.rbegin();
  }
  auto rend() {
    return gathered.rend();
  }
  auto rbegin() const {
    return gathered.rbegin();
  }
  auto rend() const {
    return gathered.rend();
  }

private:
  /**
   * Gathered argument-parameter pairs.
   */
  std::list<std::pair<Expression*,VarParameter*>> gathered;
};
}
