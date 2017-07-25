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
      state(0),
      nstates(0) {
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
  virtual bool run() = 0;

  /**
   * Get the last yield value.
   */
  Type& getValue() {
    return value;
  }
  const Type& getValue() const {
    return value;
  }

protected:
  /**
   * Last yielded value.
   */
  Type value;

  /**
   * State.
   */
  int state;

  /**
   * Number of states.
   */
  int nstates;
};
}
