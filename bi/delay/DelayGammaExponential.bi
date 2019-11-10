/*
 * Delayed gamma-Exponential random variate.
 */
final class DelayGammaExponential(future:Real?, futureUpdate:Boolean,
    λ:DelayGamma) < DelayValue<Real>(future, futureUpdate) {
  /**
   * Rate.
   */
  λ:DelayGamma& <- λ;

  function simulate() -> Real {
    return simulate_lomax(1.0/λ.θ, λ.k);
  }

  function logpdf(x:Real) -> Real {
    return logpdf_lomax(x, 1.0/λ.θ, λ.k);
  }

  function update(x:Real) {
    (λ.k, λ.θ) <- update_gamma_exponential(x, λ.k, λ.θ);
  }

  function downdate(x:Real) {
    (λ.k, λ.θ) <- downdate_gamma_exponential(x, λ.k, λ.θ);
  }

  function cdf(x:Real) -> Real? {
    return cdf_lomax(x, 1.0/λ.θ, λ.k);
  }

  function quantile(p:Real) -> Real? {
    return quantile_lomax(p, 1.0/λ.θ, λ.k);
  }

  function lower() -> Real? {
    return 0.0;
  }
}

function DelayGammaExponential(future:Real?, futureUpdate:Boolean,
    λ:DelayGamma) -> DelayGammaExponential {
  m:DelayGammaExponential(future, futureUpdate, λ);
  λ.setChild(m);
  return m;
}
