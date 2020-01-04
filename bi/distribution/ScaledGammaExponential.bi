/*
 * ed scaled gamma-exponential random variate.
 */
class ScaledGammaExponential(future:Real?, futureUpdate:Boolean, a:Real,
    λ:Gamma) < Distribution<Real>(future, futureUpdate) {
  /**
   * Scale.
   */
  a:Real <- a;

  /**
   * Rate.
   */
  λ:Gamma& <- λ;

  function simulate() -> Real {
    return simulate_lomax(1.0/(a*λ.θ), λ.k);
  }

  function logpdf(x:Real) -> Real {
    return logpdf_lomax(x, 1.0/(a*λ.θ), λ.k);
  }

  function update(x:Real) {
    (λ.k, λ.θ) <- update_scaled_gamma_exponential(x, a, λ.k, λ.θ);
  }
  
  function downdate(x:Real) {
    (λ.k, λ.θ) <- downdate_scaled_gamma_exponential(x, a, λ.k, λ.θ);
  }

  function cdf(x:Real) -> Real? {
    return cdf_lomax(x, 1.0/(a*λ.θ), λ.k);
  }

  function quantile(p:Real) -> Real? {
    return quantile_lomax(p, 1.0/(a*λ.θ), λ.k);
  }

  function lower() -> Real? {
    return 0.0;
  }
}

function ScaledGammaExponential(future:Real?, futureUpdate:Boolean,
    a:Real, λ:Gamma) -> ScaledGammaExponential {
  assert a > 0;
  m:ScaledGammaExponential(future, futureUpdate, a, λ);
  λ.setChild(m);
  return m;
}
