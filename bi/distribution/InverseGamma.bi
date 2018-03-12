/**
 * Inverse-gamma distribution.
 */
class InverseGamma < Random<Real> {
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
      set(simulate_inverse_gamma(k, θ));
    } else {
      setWeight(observe_inverse_gamma(value(), k, θ));
    }
  }
}

/**
 * Create inverse-gamma distribution.
 */
function InverseGamma(k:Real, θ:Real) -> InverseGamma {
  m:InverseGamma;
  m.initialize(k, θ);
  return m;
}
