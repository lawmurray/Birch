/**
 * @file
 */
#pragma once

#include "libbirch/FiberOutput.hpp"
#include "libbirch/Tuple.hpp"

namespace libbirch {
/**
 * State of a fiber.
 *
 * @ingroup libbirch
 *
 * @tparam Yield Yield type.
 * @tparam Return Return type.
 * @tparam State State type. Typically a Tuple type.
 * @tparam Resume Resume type. Typically a lambda type.
 */
template<class Yield, class Return, class State, class Resume>
class FiberState: public Any {
public:
  using class_type_ = FiberState;
  using yield_type_ = Yield;
  using return_type_ = Return;

  /**
   * Constructor.
   */
  FiberState(Label* context, const State& state, const Resume& resume) :
      FiberOutput(context),
      state(state),
      resume(resume) {
    //
  }

  /**
   * Deep copy constructor for value yield type.
   */
  template<IS_VALUE(Yield)>
  FiberState(Label* context, Label* label, const FiberState& o) :
      FiberOutput(context, label, o),
      state(o.state),
      resume(o.resume) {
    //
  }

  /**
   * Deep copy constructor for non-value yield type.
   */
  template<IS_NOT_VALUE(Yield)>
  FiberState(Label* context, Label* label, const FiberState& o) :
      FiberOutput(context, label, o),
      state(o.state),
      resume(o.resume) {
    //
  }

  /**
   * Destructor.
   */
  virtual ~FiberState() {
    //
  }

protected:
  virtual void doFreeze_() {
    freeze(state);
    freeze(resume);
  }

  virtual void doThaw_(Label* label) {
    thaw(state, label);
    thaw(resume, label);
  }

  virtual void doFinish_() {
    finish(state);
    finish(resume);
  }

  /**
   * State.
   */
  State state;

  /**
   * Resume function.
   */
  Resume resume;
};
}
