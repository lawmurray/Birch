/*
 * Delayed Inverse-gamma gamma random variate.
 */
class DelayInverseGammaGamma(x:Random<Real>&, k:Real, θ:DelayInverseGamma) < DelayValue<Real>(x) {

  /**
   * Shape.
   */
  k:Real <- k;

  /**
   * Scale.
   */
  θ:DelayInverseGamma <- θ;

  function simulate() -> Real {
    return simulate_compound_gamma(k, θ.α, θ.β);
  }

  function observe(x:Real) -> Real {
    return observe_compound_gamma(x, k, θ.α, θ.β);
  }

  function condition(x:Real) {
    θ.α <- θ.α + k;
    θ.β <- θ.β + x;
  }

  function pdf(x:Real) -> Real {
    return pdf_compound_gamma(x, k, θ.α, θ.β);
  }

  function cdf(x:Real) -> Real {
    return cdf_compound_gamma(x, k, θ.α, θ.β);
  }

  function lower() -> Real? {
    return 0.0;
  }
}

function DelayInverseGammaGamma(x:Random<Real>&, k:Real, θ:DelayInverseGamma) ->
    DelayInverseGammaGamma {
  m:DelayInverseGammaGamma(x, k, θ);
  θ.setChild(m);
  return m;
}
