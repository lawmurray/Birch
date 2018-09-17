/**
 * @file
 */
#pragma once

#include "libbirch/Any.hpp"

namespace bi {
/**
 * State of a fiber.
 *
 * @ingroup libbirch
 *
 * @tparam YieldType Yield type.
 */
template<class YieldType>
class FiberState: public Any {
public:
  using yield_type = YieldType;

  /**
   * Constructor.
   *
   * @param label Initial label.
   * @param nlabels Number of labels.
   */
  FiberState(const int label = 0, const int nlabels = 0) :
      label(label),
      nlabels(nlabels) {
    //
  }

  /**
   * Copy constructor.
   */
  FiberState(const FiberState<YieldType>& o) :
      label(o.label),
      nlabels(o.nlabels),
      value(o.value) {
    //
  }

  /**
   * Copy assignment.
   */
  FiberState<YieldType>& operator=(const FiberState<YieldType>& o) {
    label = o.label;
    nlabels = o.nlabels;
    value = o.value;
    return *this;
  }

  /**
   * Destructor.
   */
  virtual ~FiberState() {
    //
  }

  virtual void destroy() {
    this->size = sizeof(*this);
    this->~FiberState();
  }

  /**
   * Run to next yield point.
   */
  virtual bool query() = 0;

  /**
   * Get the last yield value.
   */
  YieldType& get() {
    return value;
  }

protected:
  /**
   * Current label.
   */
  int label;

  /**
   * Number of labels.
   */
  int nlabels;

  /**
   * Yield value.
   */
  YieldType value;
};
}
