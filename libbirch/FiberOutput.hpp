/**
 * @file
 */
#pragma once

#include "libbirch/Any.hpp"

namespace libbirch {
/**
 * Output of a fiber.
 *
 * @ingroup libbirch
 *
 * @tparam Yield Yield type.
 * @tparam Return Return type.
 */
template<class Yield, class Return>
class FiberOutput: public Any {
public:
  using class_type_ = FiberOutput<Yield,Return>;
  using yield_type_ = Yield;
  using return_type_ = Return;

  /**
   * Constructor.
   */
  FiberOutput(Label* context) :
      Any(context) {
    //
  }

  /**
   * Deep copy constructor for value yield type.
   */
  template<IS_VALUE(Yield)>
  FiberOutput(Label* context, Label* label, const FiberOutput& o) :
      Any(context, label, o),
      yieldValue(o.yieldValue),
      returnValue(o.returnValue) {
    //
  }

  /**
   * Deep copy constructor for non-value yield type.
   */
  template<IS_NOT_VALUE(Yield)>
  FiberOutput(Label* context, Label* label, const FiberOutput& o) :
      Any(context, label, o),
      yieldValue(context, label, o.yieldValue),
      returnValue(context, label, o.returnValue) {
    //
  }

  /**
   * Destructor.
   */
  virtual ~FiberOutput() {
    //
  }

  /**
   * Run to next yield point.
   */
  virtual bool query() = 0;

  /**
   * Get the last yield value.
   */
  Yield& get() {
    return yieldValue;
  }

protected:
  virtual void doFreeze_() {
    freeze(yieldValue);
    freeze(returnValue);
  }

  virtual void doThaw_(Label* label) {
    thaw(yieldValue, label);
    thaw(returnValue, label);
  }

  virtual void doFinish_() {
    finish(yieldValue);
    finish(returnValue);
  }

  /**
   * Most recent yield value.
   */
  Yield yieldValue;

  /**
   * Return value.
   */
  Return returnValue;
};
}
