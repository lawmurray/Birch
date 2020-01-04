/*
 * ed scaled gamma-Poisson random variate.
 */
final class ScaledGammaPoisson(future:Integer?, futureUpdate:Boolean,
    a:Real, λ:Gamma) < Discrete(future, futureUpdate) {
  /**
   * Scale.
   */
  a:Real <- a;

  /**
   * Rate.
   */
  λ:Gamma& <- λ;

  function simulate() -> Integer {
    if value? {
      return value!;
    } else {
      return simulate_gamma_poisson(λ.k, a*λ.θ);
    }
  }
  
  function logpdf(x:Integer) -> Real {
    return logpdf_gamma_poisson(x, λ.k, a*λ.θ);
  }

  function update(x:Integer) {
    (λ.k, λ.θ) <- update_scaled_gamma_poisson(x, a, λ.k, λ.θ);
  }

  function downdate(x:Integer) {
    (λ.k, λ.θ) <- downdate_scaled_gamma_poisson(x, a, λ.k, λ.θ);
  }

  function cdf(x:Integer) -> Real? {
    return cdf_gamma_poisson(x, λ.k, a*λ.θ);
  }

  function quantile(P:Real) -> Integer? {
    return quantile_gamma_poisson(P, λ.k, a*λ.θ);
  }

  function lower() -> Integer? {
    return 0;
  }
}

function ScaledGammaPoisson(future:Integer?, futureUpdate:Boolean,
    a:Real, λ:Gamma) -> ScaledGammaPoisson {
  assert a > 0;
  m:ScaledGammaPoisson(future, futureUpdate, a, λ);
  λ.setChild(m);
  return m;
}
