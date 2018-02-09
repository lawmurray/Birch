/**
 * @file
 */
#pragma once

namespace bi {
/**
 * State of a fiber.
 *
 * @ingroup libbirch
 *
 * @tparam YieldType Yield type.
 */
template<class YieldType>
class FiberState {
public:
  /**
   * Constructor.
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

  /**
   * Clone the object.
   */
  virtual std::shared_ptr<FiberState<YieldType>> clone() const = 0;

  /**
   * Get the world in which the fiber runs.
   */
  virtual const std::shared_ptr<World>& getWorld() = 0;

  /**
   * Run to next yield point.
   */
  virtual bool query() = 0;

  /**
   * Get the last yield value.
   *
   * @internal Returns by value to ensure that pointers, from the fiber's
   * world, are mapped to the caller's world.
   */
  const YieldType get() const {
    return value;
  }

protected:
  /**
   * Last yielded value.
   */
  YieldType value;

  /**
   * Label to which to jump on next query.
   */
  int label;

  /**
   * Number of labels.
   */
  int nlabels;
};
}
