/**
 * @file
 */
#pragma once

#include "libbirch/FiberState.hpp"
#include "libbirch/Optional.hpp"
#include "libbirch/Lazy.hpp"
#include "libbirch/SharedPtr.hpp"

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
class Fiber {
public:
  using yield_type = Yield;
  using return_type = Return;
  using state_type = Lazy<SharedPtr<FiberState<Yield,Return>>>;

  /**
   * Constructor. Used:
   *
   * @li for an uninitialized fiber handle, and
   * @li for returns in fibers with a return type of `void`, where no state
   * or resume function is required, and no value is returned.
   */
  Fiber() {
    //
  }

  /**
   * Constructor. Used:
   *
   * @li in the initialization function of all fibers, where a state and
   * start function are required, but no value is yielded, and
   * @li for yields in fibers with a yield type of `void`, where a state and
   * resume function are required, but no value is yielded.
   */
  Fiber(const state_type& state) :
      state(state) {
    //
  }

  /**
   * Constructor. Used for yields in fibers with a yield type that is not
   * `void`, where a state and resume function are required, along with a
   * yield value.
   */
  Fiber(const yield_type& yieldValue, const state_type& state) :
      yieldValue(yieldValue),
      state(state) {
    //
  }

  /**
   * Constructor. Used for returns in fibers with a return type that is not
   * `void`, where a state and resume function are not required, and a value
   * is returned.
   */
  Fiber(const return_type& returnValue) :
      returnValue(returnValue) {
    //
  }

  /**
   * Relabel.
   */
  void relabel(Label* oldLabel, Label* newLabel);

  /**
   * Run to next yield point.
   *
   * @return Was a value yielded?
   */
  bool query() {
    if (state.query()) {
      *this = state.get()->query();
    }
    return state.query();
  }

  /**
   * Get the current yield value.
   */
  auto get() {
    return yieldValue.get();
  }

private:
  /**
   * Yield value.
   */
  Optional<yield_type> yieldValue;

  /**
   * Return value.
   */
  Optional<return_type> returnValue;

  /**
   * Fiber state.
   */
  Optional<state_type> state;
};

template<class Return>
class Fiber<void,Return> {
public:
  using yield_type = void;
  using return_type = Return;
  using state_type = Lazy<SharedPtr<FiberState<yield_type,return_type>>>;

  Fiber() {
    //
  }

  Fiber(const state_type& state) :
      state(state) {
    //
  }

  Fiber(const return_type& returnValue) :
      returnValue(returnValue) {
    //
  }

  void relabel(Label* oldLabel, Label* newLabel);

  bool query() {
    if (state.query()) {
      *this = state.get()->query();
    }
    return state.query();
  }

private:
  Optional<return_type> returnValue;
  Optional<state_type> state;
};

template<class Yield>
class Fiber<Yield,void> {
public:
  using yield_type = Yield;
  using return_type = void;
  using state_type = Lazy<SharedPtr<FiberState<yield_type,return_type>>>;

  Fiber() {
    //
  }

  Fiber(const state_type& state) :
      state(state) {
    //
  }

  Fiber(const yield_type& yieldValue, const state_type& state) :
      yieldValue(yieldValue),
      state(state) {
    //
  }

  void relabel(Label* oldLabel, Label* newLabel);

  bool query() {
    if (state.query()) {
      *this = state.get()->query();
    }
    return state.query();
  }

  auto get() {
    return yieldValue.get();
  }

private:
  Optional<yield_type> yieldValue;
  Optional<state_type> state;
};

template<>
class Fiber<void,void> {
public:
  using yield_type = void;
  using return_type = void;
  using state_type = Lazy<SharedPtr<FiberState<yield_type,return_type>>>;

  Fiber() {
    //
  }

  Fiber(const state_type& state) :
      state(state) {
    //
  }

  void relabel(Label* oldLabel, Label* newLabel);

  bool query() {
    if (state.query()) {
      *this = state.get()->query();
    }
    return state.query();
  }

private:
  Optional<state_type> state;
};

template<class Yield, class Return>
struct is_value<Fiber<Yield,Return>> {
  static const bool value = false;
};

template<class Yield, class Return>
void relabel(Label* oldLabel, Label* newLabel, Fiber<Yield,Return>& o) {
  o.relabel(oldLabel, newLabel);
}
}

#include "libbirch/relabel.hpp"

template<class Yield, class Return>
void libbirch::Fiber<Yield,Return>::relabel(Label* oldLabel,
    Label* newLabel) {
  libbirch::relabel(oldLabel, newLabel, yieldValue);
  libbirch::relabel(oldLabel, newLabel, returnValue);
  libbirch::relabel(oldLabel, newLabel, state);
}

template<class Return>
void libbirch::Fiber<void,Return>::relabel(Label* oldLabel,
    Label* newLabel) {
  libbirch::relabel(oldLabel, newLabel, returnValue);
  libbirch::relabel(oldLabel, newLabel, state);
}

template<class Yield>
void libbirch::Fiber<Yield,void>::relabel(Label* oldLabel,
    Label* newLabel) {
  libbirch::relabel(oldLabel, newLabel, yieldValue);
  libbirch::relabel(oldLabel, newLabel, state);
}

inline void libbirch::Fiber<void,void>::relabel(Label* oldLabel,
    Label* newLabel) {
  libbirch::relabel(oldLabel, newLabel, state);
}
