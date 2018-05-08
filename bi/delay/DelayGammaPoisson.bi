/*
 * Delayed gamma-Poisson random variate.
 */
class DelayGammaPoisson(λ:DelayGamma) < DelayValue<Integer> {
  /**
   * Rate.
   */
  λ:DelayGamma <- λ;

  function simulate() -> Integer {
    return simulate_gamma_poisson(λ.k, λ.θ);
  }
  
  function observe(x:Integer) -> Real {
    return observe_gamma_poisson(x, λ.k, λ.θ);
  }

  function condition(x:Integer) {
    (λ.k, λ.θ) <- update_gamma_poisson(x, λ.k, λ.θ);
  }

  function pmf(x:Integer) -> Real {
    return pmf_gamma_poisson(x, λ.k, λ.θ);
  }

  function cdf(x:Integer) -> Real {
    return cdf_gamma_poisson(x, λ.k, λ.θ);
  }
}

function DelayGammaPoisson(λ:DelayGamma) ->  DelayGammaPoisson {
  m:DelayGammaPoisson(λ);
  return m;
}
