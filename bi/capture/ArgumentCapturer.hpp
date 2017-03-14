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

private:
  /**
   * Gathered argument-parameter pairs.
   */
  std::list<std::pair<Expression*,VarParameter*>> gathered;
};
}
