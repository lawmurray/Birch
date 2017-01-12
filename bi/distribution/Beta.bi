import math;
import random;
import assert;
import distribution.Gamma;
import distribution.Bernoulli;

/**
 * Beta distribution.
 */
model Beta {
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
function Beta(α:Real, β:Real) -> m:Beta {
  m.α <- α;
  m.β <- β;
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
 * Evaluate pdf.
 */
function x:Real ~> m:Beta -> l:Real {
  l <- exp(log(x ~> m));
}

/**
 * Evaluate log-pdf.
 */
function log(x:Real ~> m:Beta) -> ll:Real {
  /* pre-condition */
  require(0.0 < x && x < 1.0);

  logZ:Real <- lgamma(m.α) + lgamma(m.β) - lgamma(m.α + m.β);
  ll <- (m.α - 1.0)*log(x) + (m.β - 1.0)*log(1.0 - x) - logZ;
}
