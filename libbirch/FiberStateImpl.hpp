/**
 * @file
 */
#pragma once

#include "libbirch/FiberState.hpp"
#include "libbirch/Tuple.hpp"

namespace libbirch {
/**
 * Concrete state of a fiber.
 *
 * @ingroup libbirch
 *
 * @tparam Yield Yield type.
 * @tparam Return Return type.
 * @tparam Resume Resume type. Typically a lambda type.
 * @tparam State State type. Typically a tuple type.
 */
template<class Yield, class Return, class Resume, class State>
class FiberStateImpl: public FiberState<Yield,Return> {
public:
  using class_type_ = FiberStateImpl;

  /**
   * Constructor.
   */
  FiberStateImpl(const Resume& resume, const State& state) :
      FiberState<Yield,Return>(),
      resume(resume),
      state(state) {
    //
  }

  /**
   * Destructor.
   */
  virtual ~FiberStateImpl() {
    //
  }

  /**
   * Resume.
   */
  virtual Fiber<Yield,Return> query() {
    return resume(state);
  }

protected:
  /**
   * Resume function.
   */
  Resume resume;

  /**
   * State tuple.
   */
  State state;
};
}
