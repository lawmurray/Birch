/**
 * @file
 */
#pragma once

#include "libbirch/Any.hpp"

namespace libbirch {
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

  /**
   * Constructor.
   *
   * @param npoints Number of yield points.
   */
  FiberState(Label* context, const int npoints) :
      Any(context),
      point_(0),
      npoints_(npoints) {
    //
  }

  /**
   * Deep copy constructor.
   */
  FiberState(Label* label, const FiberState<YieldType>& o) :
      Any(label, o),
      point_(o.point_),
      npoints_(o.npoints_) {
    //
  }

  /**
   * Destructor.
   */
  virtual ~FiberState() {
    //
  }

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
   * Current yield point.
   */
  int point_;

  /**
   * Number of yield points.
   */
  int npoints_;
};
}
