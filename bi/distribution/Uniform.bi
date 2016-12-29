import math;
import random;

/**
 * Uniform distribution.
 */
model Uniform(l1:Real, u1:Real) {
  /**
   * Lower bound.
   */
  l:Real <- l1;
  
  /**
   * Upper bound.
   */
  u:Real <- u1;
}

/**
 * Simulate.
 */
function m:Uniform ~> x:Real {
  cpp {{
  x = std::uniform_real_distribution<double>(m.l, m.u)(rng);
  }}
}

/**
 * Evaluate pdf.
 */
function x:Real ~ m:Uniform -> l:Real {
  if (x >= m.l && x <= m.u) {
    l <- 1.0/(m.u - m.l);
  } else {
    l <- 0.0;
  }
}
