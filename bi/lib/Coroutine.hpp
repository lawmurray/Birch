/**
 * @file
 */
#pragma once

#include "bi/lib/Object.hpp"

namespace bi {
/**
 * Coroutine.
 *
 * @ingroup library
 *
 * @tparam Type Return type.
 */
template<class Type>
class Coroutine: public Object {
public:
  /**
   * Constructor.
   */
  Coroutine() :
      state(0) {
    //
  }

  /**
   * Destructor.
   */
  virtual ~Coroutine() {
    //
  }

  /**
   * Clone the object.
   */
  virtual Coroutine<Type>* clone() = 0;

  /**
   * Run to next yield point.
   */
  virtual Type operator()() = 0;

protected:
  /**
   * State.
   */
  int state;
};
}
