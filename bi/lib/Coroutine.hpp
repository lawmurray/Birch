/**
 * @file
 */
#pragma once

namespace bi {
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
   * Caller.
   */
  virtual Type operator()() = 0;

protected:
  /**
   * State.
   */
  int state;
};
}
