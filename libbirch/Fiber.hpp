/**
 * @file
 */
#pragma once

#include "libbirch/FiberYield.hpp"
#include "libbirch/FiberReturn.hpp"
#include "libbirch/FiberState.hpp"
#include "libbirch/FiberStateImpl.hpp"

namespace libbirch {
/**
 * Fiber.
 *
 * @ingroup libbirch
 *
 * @tparam Yield Yield type.
 * @tparam Return Return type.
 */
template<class Yield, class Return>
class Fiber : public FiberYield<Yield>, public FiberReturn<Return> {
public:
  using yield_type = Yield;
  using return_type = Return;
  using state_type = Lazy<SharedPtr<FiberState<Yield,Return>>>;

  /**
   * Constructor.
   */
  Fiber() {
    //
  }

  /**
   * Constructor.
   */
  Fiber(Label* context) {
    //
  }

  /**
   * Constructor.
   */
  Fiber(Label* context, const state_type& state) :
      state(context, state) {
    //
  }

  /**
   * Constructor.
   */
  template<class T, std::enable_if_t<std::is_same<T,yield_type>::value && !std::is_void<yield_type>::value,int> = 0>
  Fiber(Label* context, const T& yieldValue, const state_type& state) :
      FiberYield<Yield>(context, yieldValue),
      state(context, state) {
    //
  }

  /**
   * Constructor.
   */
  template<class T, std::enable_if_t<std::is_same<T,return_type>::value && !std::is_void<return_type>::value,int> = 0>
  Fiber(Label* context, const T& returnValue) :
      FiberReturn<Return>(context, returnValue) {
    //
  }

  /**
   * Copy constructor.
   */
  Fiber(Label* context, const Fiber& o) :
      FiberYield<Yield>(context, o),
      FiberReturn<Return>(context, o),
      state(context, o.state) {
    //
  }

  /**
   * Move constructor.
   */
  Fiber(Label* context, Fiber&& o) :
      FiberYield<Yield>(context, std::move(o)),
      FiberReturn<Return>(context, std::move(o)),
      state(context, std::move(o.state)) {
    //
  }

  /**
   * Deep copy constructor.
   */
  Fiber(Label* context, Label* label, const Fiber& o) :
      FiberYield<Yield>(context, label, o),
      FiberReturn<Return>(context, label, o),
      state(context, label, o.state) {
    //
  }

  /**
   * Copy assignment.
   */
  Fiber& assign(Label* context, const Fiber& o) {
    FiberYield<Yield>::assign(context, o);
    FiberReturn<Return>::assign(context, o);
    state.assign(context, o.state);
    return *this;
  }

  /**
   * Move assignment.
   */
  Fiber& assign(Label* context, Fiber&& o) {
    FiberYield<Yield>::assign(context, std::move(o));
    FiberReturn<Return>::assign(context, std::move(o));
    state.assign(context, std::move(o.state));
    return *this;
  }

  /**
   * Clone the fiber.
   */
  Fiber<Yield,Return> clone(Label* context) const {
    return Fiber(context, *this);
  }

  /**
   * Freeze the fiber.
   */
  void freeze() const {
    FiberYield<Yield>::freeze();
    FiberReturn<Return>::freeze();
    freeze(state);
  }

  /**
   * Thaw the fiber.
   */
  void thaw(Label* label) const {
    FiberYield<Yield>::thaw(label);
    FiberReturn<Return>::thaw(label);
    thaw(state, label);
  }

  /**
   * Finish the fiber.
   */
  void finish() const {
    FiberYield<Yield>::finish();
    FiberReturn<Return>::finish();
    finish(state);
  }

  /**
   * Run to next yield point.
   *
   * @return Was a value yielded?
   */
  bool query() {
    if (state.query()) {
      *this = state.get()->query();
      return state.query();
    } else {
      return false;
    }
  }

private:
  /**
   * Fiber state.
   */
  Optional<state_type> state;
};

template<class Yield, class Return>
struct is_value<Fiber<Yield,Return>> {
  static const bool value = false;
};
}
