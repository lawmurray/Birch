/**
 * @file
 */
#pragma once

#include "bi/lib/Object.hpp"

namespace bi {
/**
 * Fiber state.
 *
 * @ingroup library
 *
 * @tparam Type Return type.
 */
template<class Type>
class FiberState: public Object {
public:
  typedef FiberState<Type> this_type;
  typedef Object super_type;

  /**
   * Constructor.
   */
  FiberState() :
      label(0),
      nlabels(0) {
    //
  }

  /**
   * Destructor.
   */
  virtual ~FiberState() {
    //
  }

  /**
   * Clone the object.
   */
  virtual FiberState<Type>* clone() = 0;

  /**
   * Run to next yield point.
   */
  virtual bool query() = 0;

  /**
   * Get the last yield value.
   */
  Type& get() {
    return value;
  }
  const Type& get() const {
    return value;
  }

protected:
  /**
   * Last yielded value.
   */
  Type value;

  /**
   * Label to which to jump on next query.
   */
  int label;

  /**
   * Number of labels.
   */
  int nlabels;
};
}
