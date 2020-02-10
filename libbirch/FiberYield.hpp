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
  FiberYield() {
    //
  }

  /**
   * Constructor.
   */
  FiberYield(Label* context) {
    //
  }

  /**
   * Constructor.
   */
  FiberYield(Label* context, const yield_type& yieldValue) :
      yieldValue(context, yieldValue) {
    //
  }

  /**
   * Copy constructor.
   */
  FiberYield(Label* context, const FiberYield& o) :
      yieldValue(context, o.yieldValue) {
    //
  }

  /**
   * Move constructor.
   */
  FiberYield(Label* context, FiberYield&& o) :
      yieldValue(context, std::move(o.yieldValue)) {
    //
  }

  /**
   * Deep copy constructor.
   */
  FiberYield(Label* context, Label* label, const FiberYield& o) :
      yieldValue(context, label, o.yieldValue) {
    //
  }

  /**
   * Copy assignment.
   */
  FiberYield& assign(Label* context, const FiberYield& o) {
    yieldValue.assign(context, o.yieldValue);
    return *this;
  }

  /**
   * Move assignment.
   */
  FiberYield& assign(Label* context, FiberYield&& o) {
    yieldValue.assign(context, std::move(o.yieldValue));
    return *this;
  }

  /**
   * Get yield value.
   */
  yield_type get() const {
    return yieldValue.get();
  }

  void freeze() const {
    freeze(yieldValue);
  }

  void thaw(Label* label) const {
    thaw(yieldValue, label);
  }

  void finish() const {
    finish(yieldValue);
  }

private:
  /**
   * Yield value.
   */
  Optional<yield_type> yieldValue;
};

template<class Yield>
class FiberYield<Yield,IS_VOID(Yield)> {
  using yield_type = Yield;

  /**
   * Constructor.
   */
  FiberYield() {
    //
  }

  /**
   * Constructor.
   */
  FiberYield(Label* context) {
    //
  }

  /**
   * Copy constructor.
   */
  FiberYield(Label* context, const FiberYield& o) {
    //
  }

  /**
   * Move constructor.
   */
  FiberYield(Label* context, FiberYield&& o) {
    //
  }

  /**
   * Deep copy constructor.
   */
  FiberYield(Label* context, Label* label, const FiberYield& o) {
    //
  }

  /**
   * Copy assignment.
   */
  FiberYield& assign(Label* context, const FiberYield& o) {
    return *this;
  }

  /**
   * Move assignment.
   */
  FiberYield& assign(Label* context, FiberYield&& o) {
    return *this;
  }

  void freeze() const {
    //
  }

  void thaw(Label* label) const {
    //
  }

  void finish() const {
    //
  }
};
}
