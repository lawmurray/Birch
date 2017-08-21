import delay.DelayReal;
import math;
import random;

/**
 * Uniform distribution.
 */
class Uniform < DelayReal {
  /**
   * Lower bound.
   */
  l:Real;
  
  /**
   * Upper bound.
   */
  u:Real;

  function initialize(l:Real, u:Real) {
    assert l <= u;
  
    super.initialize();
    this.l <- l;
    this.u <- u;
  }

  function update(l:Real, u:Real) {
    assert l <= u;
  
    this.l <- l;
    this.u <- u;
  }

  function doRealize() {
    if (isMissing()) {
      set(random_uniform(l, u));
    } else {
      if (x >= l && x <= u) {
        setWeight(log(1.0/(u - l)));
      } else {
        setWeight(-inf);
      }
    }
  }
}

/**
 * Create a Uniform distribution.
 */
function Uniform(l:Real, u:Real) -> Uniform {
  m:Uniform;
  m.initialize(l, u);
  return m;
}
