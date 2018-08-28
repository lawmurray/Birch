/*
 * Delayed gamma-Exponential random variate.
 */
class DelayGammaExponential(x:Random<Real>&, λ:DelayGamma) <
    DelayValue<Real>(x) {
  /**
   * Rate.
   */
  λ:DelayGamma <- λ;

  function simulate() -> Real {
    return simulate_lomax(1.0/λ.θ, λ.k);
  }

  function observe(x:Real) -> Real {
    return observe_lomax(x, 1.0/λ.θ, λ.k);
  }

  function condition(x:Real) {
    λ.k <- λ.k + 1.0;
    λ.θ <- λ.θ / (1.0 + x*λ.θ);
  }

  function pdf(x:Real) -> Real {
    return pdf_lomax(x, 1.0/λ.θ, λ.k);
  }

  function cdf(x:Real) -> Real {
    return cdf_lomax(x, 1.0/λ.θ, λ.k);
  }

  function lower() -> Real? {
    return 0.0;
  }
}

function DelayGammaExponential(x:Random<Real>&, λ:DelayGamma) ->
    DelayGammaExponential {
  m:DelayGammaExponential(x, λ);
  λ.setChild(m);
  return m;
}
