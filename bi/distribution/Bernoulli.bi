import math;
import random;

/**
 * Bernoulli distribution.
 */
class Bernoulli {
  /**
   * Probability of a true result.
   */
  ρ:Real;

  /**
   * Simulate.
   */
  function simulate() -> Boolean {
    cpp {{
    return std::bernoulli_distribution(ρ_)(rng);
    }}
  }

  /**
   * Observe.
   */
  function observe(x:Boolean) -> Real {
    if (x) {
      return log(ρ);
    } else {
      return log(1.0 - ρ);
    }
  }
}

/**
 * Create.
 */
function Bernoulli(ρ:Real) -> Bernoulli {
  m:Bernoulli;
  m.ρ <- ρ;
  return m;
}
