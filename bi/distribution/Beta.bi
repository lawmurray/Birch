import math;
import random;
import distribution.Gamma;
import distribution.Bernoulli;

/**
 * Beta distribution.
 */
class Beta {
  /**
   * First shape parameter.
   */
  α:Real;

  /**
   * Second shape parameter.
   */
  β:Real;

  /**
   * Simulate.
   */
  function simulate() -> Real {
    u:Real;
    v:Real;
    u <~ Gamma(α, 1.0);
    v <~ Gamma(β, 1.0);
    return u/(u + v);
  }

  /**
   * Observe.
   */
  function observe(x:Real) -> Real {
    if (0.0 < x && x < 1.0) {
      logZ:Real <- lgamma(α) + lgamma(β) - lgamma(α + β);
      return (α - 1.0)*log(x) + (β - 1.0)*log(1.0 - x) - logZ;
    } else {
      return log(0.0);
    }
  }
}

/**
 * Create.
 */
function Beta(α:Real, β:Real) -> Beta {
  m:Beta;
  m.α <- α;
  m.β <- β;
  return m;
}
