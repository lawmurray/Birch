import math;
import random;

/**
 * Gamma distribution.
 */
class Gamma {
  /**
   * Shape.
   */
  k:Real;
  
  /**
   * Scale.
   */
  θ:Real;

  /**
   * Simulate.
   */
  function simulate() -> Real {
    cpp {{
    return std::gamma_distribution<double>(k_, θ_)(rng);
    }}
  }

  /**
   * Observe.
   */
  function observe(x:Real) -> Real {
    return (k - 1.0)*log(x) - x/θ - lgamma(k) - k*log(θ);
  }
}

/**
 * Create.
 */
function Gamma(k:Real, θ:Real) -> Gamma {
  m:Gamma;
  m.k <- k;
  m.θ <- θ;
  return m;
}
