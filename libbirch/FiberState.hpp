/**
 * @file
 */
#pragma once

#include "libbirch/Any.hpp"

namespace libbirch {
template<class Return, class Yield> class Fiber;

/**
 * Abstract state of a fiber.
 *
 * @ingroup libbirch
 *
 * @tparam Return Return type.
 * @tparam Yield Yield type.
 */
template<class Return, class Yield>
class FiberState: public Any {
public:
  LIBBIRCH_ABSTRACT_CLASS(FiberState, Any)
  LIBBIRCH_MEMBERS()

  /**
   * Destructor.
   */
  virtual ~FiberState() {
    //
  }

  /**
   * Run to next yield or return point.
   *
   * @return New fiber handle.
   */
  virtual Fiber<Return,Yield> query() = 0;
};
}
