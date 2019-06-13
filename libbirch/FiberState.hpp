/**
 * @file
 */
#pragma once

#include "libbirch/Any.hpp"

namespace libbirch {
#pragma omp declare target
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
  using class_type_ = FiberState<YieldType>;
  using yield_type_ = YieldType;

protected:
  /**
   * Constructor.
   *
   * @param label Initial label.
   * @param nlabels Number of labels.
   */
  FiberState(const int label = 0, const int nlabels = 0) :
      label_(label),
      nlabels_(nlabels) {
    //
  }

  /**
   * Destructor.
   */
  virtual ~FiberState() {
    //
  }

public:
  /**
   * Run to next yield point.
   */
  virtual bool query() = 0;

  /**
   * Get the last yield value.
   */
  YieldType& get() {
    return value_;
  }

protected:
  /**
   * Current label.
   */
  int label_;

  /**
   * Number of labels.
   */
  int nlabels_;

  /**
   * Yield value.
   */
  YieldType value_;
};
#pragma omp end declare target
}
