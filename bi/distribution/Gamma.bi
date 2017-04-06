import math;
import random;
import assert;

/**
 * Gamma distribution.
 */
model Gamma {
  /**
   * Shape.
   */
  k:Real;
  
  /**
   * Scale.
   */
  θ:Real;
}

/**
 * Create.
 */
function Gamma(k:Real, θ:Real) -> m:Gamma {
  m.k <- k;
  m.θ <- θ;
}

/**
 * Simulate.
 */
function (x:Real <~ m:Gamma) {
  cpp {{
  x = std::gamma_distribution<double>(m.k, m.θ)(rng);
  }}
}

/**
 * Observe.
 */
function (x:Real ~> m:Gamma) -> l:Real {
  logZ:Real <- lgamma(m.k) + m.k*log(m.θ);
  l <- (m.k - 1.0)*log(x) - x/m.θ - logZ;
}
