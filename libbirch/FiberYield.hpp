/**
 * @file
 */
#pragma once

namespace libbirch {
/**
 * Fiber yield value.
 *
 * @ingroup libbirch
 *
 * @tparam Yield Yield type.
 */
template<class Yield, class Enable = void>
class FiberYield {
public:
  using yield_type = Yield;

  /**
   * Constructor.
   */
  FiberYield(const yield_type& yieldValue) :
      yieldValue(yieldValue) {
    //
  }

  /**
   * Get yield value.
   */
  yield_type get() const {
    return yieldValue.get();
  }

private:
  /**
   * Yield value.
   */
  Optional<yield_type> yieldValue;
};

template<class Yield>
class FiberYield<Yield,IS_VOID(Yield)> {
public:
  using yield_type = Yield;
};
}
