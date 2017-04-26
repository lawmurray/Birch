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
}

/**
 * Create.
 */
function Uniform(l:Real, u:Real) -> m:Uniform {
  m.l <- l;
  m.u <- u;
}

/**
 * Simulate.
 */
function (x:Real <~ m:Uniform) {
  cpp {{
  x = std::uniform_real_distribution<double>(m->l, m->u)(rng);
  }}
}

/**
 * Observe.
 */
function (x:Real ~> m:Uniform) -> l:Real {
  if (x >= m.l && x <= m.u) {
    l <- log(1.0/(m.u - m.l));
  } else {
    l <- log(0.0);
  }
}
