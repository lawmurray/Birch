/*
 * ed gamma-Poisson random variate.
 */
final class GammaPoisson(future:Integer?, futureUpdate:Boolean,
    λ:Gamma) < Discrete(future, futureUpdate) {
  /**
   * Rate.
   */
  λ:Gamma& <- λ;

  function simulate() -> Integer {
    if value? {
      return value!;
    } else {
      return simulate_gamma_poisson(λ.k, λ.θ);
    }
  }
  
  function logpdf(x:Integer) -> Real {
    return logpdf_gamma_poisson(x, λ.k, λ.θ);
  }

  function update(x:Integer) {
    (λ.k, λ.θ) <- update_gamma_poisson(x, λ.k, λ.θ);
  }

  function downdate(x:Integer) {
    (λ.k, λ.θ) <- downdate_gamma_poisson(x, λ.k, λ.θ);
  }

  function cdf(x:Integer) -> Real? {
    return cdf_gamma_poisson(x, λ.k, λ.θ);
  }

  function quantile(P:Real) -> Integer? {
    return quantile_gamma_poisson(P, λ.k, λ.θ);
  }

  function lower() -> Integer? {
    return 0;
  }
}

function GammaPoisson(future:Integer?, futureUpdate:Boolean,
    λ:Gamma) ->  GammaPoisson {
  m:GammaPoisson(future, futureUpdate, λ);
  λ.setChild(m);
  return m;
}
