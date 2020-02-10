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
  FiberStateImpl(Label* context, const Resume& resume, const State& state) :
      FiberState<Yield,Return>(context),
      resume(resume),
      state(state) {
    //
  }

  /**
   * Deep copy constructor for value yield type.
   */
  template<IS_VALUE(Yield)>
  FiberStateImpl(Label* context, Label* label, const FiberStateImpl& o) :
      FiberState<Yield,Return>(context, label, o),
      resume(o.resume),
      state(o.state) {
    //
  }

  /**
   * Deep copy constructor for non-value yield type.
   */
  template<IS_NOT_VALUE(Yield)>
  FiberStateImpl(Label* context, Label* label, const FiberStateImpl& o) :
      FiberState<Yield,Return>(context, label, o),
      resume(o.resume),
      state(o.state) {
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
  virtual void doFreeze_() {
    freeze(resume);
    freeze(state);
  }

  virtual void doThaw_(Label* label) {
    thaw(resume, label);
    thaw(state, label);
  }

  virtual void doFinish_() {
    finish(resume);
    finish(state);
  }

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
