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
   * Constructor.
   */
  FiberState(Label* context) :
      Any(context) {
    //
  }

  /**
   * Deep copy constructor for value yield type.
   */
  template<IS_VALUE(Yield)>
  FiberState(Label* context, Label* label, const FiberState& o) :
      Any(context, label, o) {
    //
  }

  /**
   * Deep copy constructor for non-value yield type.
   */
  template<IS_NOT_VALUE(Yield)>
  FiberState(Label* context, Label* label, const FiberState& o) :
      Any(context, label, o) {
    //
  }

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
