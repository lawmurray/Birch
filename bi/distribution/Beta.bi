import delay.DelayReal;
import math;

/**
 * Beta distribution.
 */
class Beta < DelayReal {
  /**
   * First shape parameter.
   */
  α:Real;

  /**
   * Second shape parameter.
   */
  β:Real;

  function initialize(α:Real, β:Real) {
    assert(α > 0.0);
    assert(β > 0.0);
  
    super.initialize();
    this.α <- α;
    this.β <- β;
  }

  function update(α:Real, β:Real) {
    assert(α > 0.0);
    assert(β > 0.0);

    this.α <- α;
    this.β <- β;
  }

  function doRealize() {
    if (isMissing()) {
      u:Real <- simulate_gamma(α, 1.0);
      v:Real <- simulate_gamma(β, 1.0);
      set(u/(u + v));
    } else {
      if (0.0 < x && x < 1.0) {
        logZ:Real <- lgamma(α) + lgamma(β) - lgamma(α + β);
        setWeight((α - 1.0)*log(x) + (β - 1.0)*log(1.0 - x) - logZ);
      } else {
        setWeight(-inf);
      }
    }
  }

  function tildeLeft() -> Beta {
    simulate();
    return this;
  }
  
  function tildeRight(left:Beta) -> Beta {
    set(left.value());
    observe();
    return this;
  }
}

/**
 * Create a Beta distribution.
 */
function Beta(α:Real, β:Real) -> Beta {
  m:Beta;
  m.initialize(α, β);
  return m;
}
