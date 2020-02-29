/**
 * @file
 */
#pragma once

#include "libbirch/Any.hpp"

namespace libbirch {
template<class Yield, class Return> class Fiber;

/**
 * Abstract state of a fiber.
 *
 * @ingroup libbirch
 *
 * @tparam Yield Yield type.
 * @tparam Return Return type.
 */
template<class Yield, class Return>
class FiberState: public Any {
public:
  using class_type_ = FiberState<Yield,Return>;

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
  virtual Fiber<Yield,Return> query() = 0;

  LIBBIRCH_ABSTRACT_CLASS(FiberState, Any)
  LIBBIRCH_MEMBERS()
};
}

namespace bi {
  namespace type {
template<class Yield, class Return>
struct super_type<libbirch::FiberState<Yield,Return>> {
  using type = libbirch::Any;
};
  }
}
