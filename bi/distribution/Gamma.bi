import math;
import random;
import assert;

/**
 * Gamma distribution.
 */
model Gamma(k1:Real, θ1:Real) {
  /**
   * Shape.
   */
  k:Real <- k1;
  
  /**
   * Scale.
   */
  θ:Real <- θ1;
}

/**
 * Simulate.
 */
function m:Gamma ~> x:Real {
  cpp {{
  x = std::gamma_distribution<double>(m.k, m.θ)(rng);
  }}
}

/**
 * Evaluate pdf.
 */
function x:Real ~ m:Gamma -> l:Real {
  l <- exp(log(x ~ m));
}

/**
 * Evaluate log-pdf.
 */
function log(x:Real ~ m:Gamma) -> ll:Real {
  logZ:Real <- lgamma(m.k) + m.k*log(m.θ);
  ll <- (m.k - 1.0)*log(x) - x/m.θ - logZ;
}
