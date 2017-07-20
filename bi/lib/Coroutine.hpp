/**
 * @file
 */
#pragma once

namespace bi {
/**
 * Global variable indicating whether the currently running code is within
 * a coroutine. This is used to flag objects as either global or
 * coroutine-local. It is incremented whenever a coroutine is resumed, and
 * decremented whenever it yields. A value greater than zero indicates that
 * the currently running code is within a coroutine.
 */
extern int inCoroutine;

/**
 * Relocatable coroutine.
 *
 * @ingroup library
 *
 * @tparam Type Return type.
 */
template<class Type>
class Coroutine {
public:
  /**
   * Constructor.
   */
  Coroutine() : state(0) {
    //
  }

  /**
   * Run to next yield.
   */
  Type operator()() {
    ++inCoroutine;
    Type result = run();
    --inCoroutine;
    return result;
  }

protected:
  /**
   * Run to next yield.
   */
  virtual Type run() = 0;

  /**
   * State.
   */
  int state;
};
}
