import math;
import random;

/**
 * Uniform distribution.
 */
class Uniform {
  /**
   * Lower bound.
   */
  l:Real;
  
  /**
   * Upper bound.
   */
  u:Real;

  /**
   * Simulate.
   */
  function simulate() -> Real {
    cpp {{
    return std::uniform_real_distribution<double>(l, u)(rng);
    }}
  }

  /**
   * Observe.
   */
  function observe(x:Real) -> Real {
    if (x >= l && x <= u) {
      return log(1.0/(u - l));
    } else {
      return log(0.0);
    }
  }
}

/**
 * Create.
 */
function Uniform(l:Real, u:Real) -> Uniform {
  m:Uniform;
  m.l <- l;
  m.u <- u;
  return m;
}
