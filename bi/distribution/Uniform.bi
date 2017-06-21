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
function Uniform(l:Real, u:Real) -> Uniform {
  m:Uniform;
  m.l <- l;
  m.u <- u;
  return m;
}

/**
 * Simulate.
 */
//operator x:Real <~ m:Uniform {
//  cpp {{
//  x = std::uniform_real_distribution<double>(m->l, m->u)(rng);
//  }}
//}

/**
 * Observe.
 */
//operator x:Real ~> m:Uniform -> Real {
//  if (x >= m.l && x <= m.u) {
//    return log(1.0/(m.u - m.l));
//  } else {
//    return log(0.0);
//  }
//}
