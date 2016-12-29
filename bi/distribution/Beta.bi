import math;
import random;
import assert;
import distribution.gamma;
import distribution.bernoulli;

/**
 * Beta distribution.
 */
model Beta(α1:Real, β1:Real) {
  /**
   * First shape parameter.
   */
  α:Real <- α1;
  
  /**
   * Second shape parameter.
   */
  β:Real <- β1;
}

/**
 * Simulate.
 */
function m:Beta ~> x:Real {
  u:Real <~ Gamma(m.α, 1.0);
  v:Real <~ Gamma(m.β, 1.0);
  x <- u/(u + v);
}

/**
 * Evaluate pdf.
 */
function x:Real ~ m:Beta -> l:Real {
  l <- exp(log(x ~ m));
}

/**
 * Evaluate log-pdf.
 */
function log(x:Real ~ m:Beta) -> ll:Real {
  /* pre-condition */
  require(0.0 < x && x < 1.0);

  logZ:Real <- lgamma(m.α) + lgamma(m.β) - lgamma(m.α + m.β);
  ll <- (m.α - 1.0)*log(x) + (m.β - 1.0)*log(1.0 - x) - logZ;
}
