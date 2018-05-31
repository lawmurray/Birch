/*
 * Delayed gamma-Poisson random variate.
 */
class DelayGammaPoisson(x:Random<Integer>&, λ:DelayGamma) <
    DelayValue<Integer>(x) {
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

  function lower() -> Integer? {
    return 0;
  }
}

function DelayGammaPoisson(x:Random<Integer>&, λ:DelayGamma) -> 
    DelayGammaPoisson {
  m:DelayGammaPoisson(x, λ);
  λ.setChild(m);
  return m;
}
