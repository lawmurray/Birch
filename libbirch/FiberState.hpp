/**
 * @file
 */
#pragma once

#include "libbirch/Counted.hpp"

#include <iostream>

namespace bi {
/**
 * State of a fiber.
 *
 * @ingroup libbirch
 *
 * @tparam YieldType Yield type.
 */
template<class YieldType>
class FiberState : public Counted {
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
   * Destructor.
   */
  virtual ~FiberState() {
    //
  }

  virtual void destroy() {
    size = sizeof(*this);
    this->~FiberState();
  }

  /**
   * Clone.
   */
  virtual FiberState<YieldType>* clone() const = 0;

  /**
   * Get the world in which the fiber runs.
   */
  virtual World* getWorld() = 0;

  /**
   * Run to next yield point.
   */
  virtual bool query() = 0;

  /**
   * Get the last yield value.
   */
  virtual YieldType& get() = 0;

protected:
  /**
   * Current label.
   */
  int label;

  /**
   * Number of labels.
   */
  int nlabels;
};
}
