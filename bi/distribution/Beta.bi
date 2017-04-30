import math;
import random;
import assert;
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

/**
 * Simulate.
 */
function x:Real <~ m:Beta {
  u:Real;
  v:Real;
  u <~ Gamma(m.α, 1.0);
  v <~ Gamma(m.β, 1.0);
  x <- u/(u + v);
}

/**
 * Observe.
 */
function x:Real ~> m:Beta -> Real {
  if (0.0 < x && x < 1.0) {
    logZ:Real <- lgamma(m.α) + lgamma(m.β) - lgamma(m.α + m.β);
    return (m.α - 1.0)*log(x) + (m.β - 1.0)*log(1.0 - x) - logZ;
  } else {
    return log(0.0);
  }
}
