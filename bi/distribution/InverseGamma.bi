/**
 * Inverse-gamma distribution.
 */
class InverseGamma < Random<Real> {
  /**
   * Shape.
   */
  α:Real;
  
  /**
   * Scale.
   */
  β:Real;

  function initialize(α:Real, β:Real) {
    super.initialize();
    update(α, β);
  }

  function update(α:Real, β:Real) {
    assert α > 0.0;
    assert β > 0.0;

    this.α <- α;
    this.β <- β;
  }

  function doRealize() {
    if (isMissing()) {
      set(simulate_inverse_gamma(α, β));
    } else {
      setWeight(observe_inverse_gamma(value(), α, β));
    }
  }
}

/**
 * Create inverse-gamma distribution.
 */
function InverseGamma(α:Real, β:Real) -> InverseGamma {
  m:InverseGamma;
  m.initialize(α, β);
  return m;
}
