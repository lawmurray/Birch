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
    super.initialize();
    update(k, θ);
  }

  function update(k:Real, θ:Real) {
    assert k > 0.0;
    assert θ > 0.0;

    this.k <- k;
    this.θ <- θ;
  }

  function doRealize() {
    if (isMissing()) {
      set(simulate_gamma(k, θ));
    } else {
      setWeight(observe_gamma(x, k, θ));
    }
  }

  function tildeLeft() -> Gamma {
    simulate();
    return this;
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
