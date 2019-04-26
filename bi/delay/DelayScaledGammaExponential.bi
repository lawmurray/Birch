/*
 * Delayed scaled gamma-Exponential random variate.
 */
class DelayScaledGammaExponential(future:Real?, futureUpdate:Boolean, a:Real,
    λ:DelayGamma) < DelayValue<Real>(future, futureUpdate) {
  /**
   * Scale.
   */
  a:Real <- a;

  /**
   * Rate.
   */
  λ:DelayGamma& <- λ;

  function simulate() -> Real {
    return simulate_lomax(1.0/a/λ!.θ, λ!.k);
  }

  function observe(x:Real) -> Real {
    return observe_lomax(x, 1.0/a/λ!.θ, λ!.k);
  }

  function condition(x:Real) {
    λ!.k <- λ!.k + 1.0;
    λ!.θ <- λ!.θ / (1.0 + x*a*λ!.θ);
  }

  function pdf(x:Real) -> Real {
    return pdf_lomax(x, 1.0/a/λ!.θ, λ!.k);
  }

  function cdf(x:Real) -> Real {
    return cdf_lomax(x, 1.0/a/λ!.θ, λ!.k);
  }

  function lower() -> Real? {
    return 0.0;
  }
}

function DelayScaledGammaExponential(future:Real?, futureUpdate:Boolean,
    a:Real, λ:DelayGamma) -> DelayScaledGammaExponential {
  assert a > 0;
  m:DelayScaledGammaExponential(future, futureUpdate, a, λ);
  λ.setChild(m);
  return m;
}
