/**
 * @file
 */
#pragma once

#include "libbirch/Counted.hpp"

namespace bi {
/**
 * State of a fiber.
 *
 * @ingroup libbirch
 *
 * @tparam YieldType Yield type.
 */
template<class YieldType>
class FiberState: public Counted {
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
      nlabels(nlabels),
      flagDirty(false) {
    //
  }

  /**
   * Copy constructor.
   */
  FiberState(const FiberState<YieldType>& o) :
      label(o.label),
      nlabels(o.nlabels),
      flagDirty(false) {
    //
  }

  /**
   * Copy assignment.
   */
  FiberState<YieldType>& operator=(const FiberState<YieldType>& o) {
    label = o.label;
    nlabels = o.nlabels;
    flagDirty = false;
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

  /**
   * Mark as dirty. When a fiber is copied, the original and the copy share
   * a state. When either is run, the state becomes dirty, and both must
   * clone the state before they run.
   */
  void dirty() {
    flagDirty = true;
  }

  /**
   * Is this dirty?
   */
  bool isDirty() const {
    return flagDirty;
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
   * Is this state dirty?
   */
  bool flagDirty;
};
}
