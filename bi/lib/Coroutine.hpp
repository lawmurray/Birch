/**
 * @file
 */
#pragma once

#include "bi/lib/Object.hpp"

namespace bi {
/**
 * Relocatable coroutine.
 *
 * @ingroup library
 *
 * @tparam Type Return type.
 */
template<class Type>
class Coroutine : public Object {
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
  virtual Type operator()() = 0;

protected:
  /**
   * State.
   */
  int state;
};
}
