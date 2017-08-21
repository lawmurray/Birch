import delay.DelayReal;
import math;
import random;

/**
 * Gamma distribution.
 */
class Gamma < DelayReal {
  /**
   * Shape.
   */
  k:Real;
  
  /**
   * Scale.
   */
  θ:Real;

  function initialize(k:Real, θ:Real) {
    assert k > 0.0;
    assert θ > 0.0;
  
    super.initialize();
    this.k <- k;
    this.θ <- θ;
  }

  function update(k:Real, θ:Real) {
    assert k > 0.0;
    assert θ > 0.0;

    this.k <- k;
    this.θ <- θ;
  }

  function doRealize() {
    if (isMissing()) {
      set(random_gamma(k, θ));
    } else {
      if (x > 0.0) {
        setWeight((k - 1.0)*log(x) - x/θ - lgamma(k) - k*log(θ));
      } else {
        setWeight(-inf);
      }
    }
  }
}

/**
 * Create Gamma distribution.
 */
function Gamma(k:Real, θ:Real) -> Gamma {
  m:Gamma;
  m.initialize(k, θ);
  return m;
}
