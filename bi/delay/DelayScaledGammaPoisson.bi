/*
 * Delayed scaled gamma-Poisson random variate.
 */
class DelayScaledGammaPoisson(x:Random<Integer>&, a:Real, λ:DelayGamma) <
    DelayDiscrete(x) {
  /**
   * Scale.
   */
  a:Real <- a;

  /**
   * Rate.
   */
  λ:DelayGamma& <- λ;

  function simulate() -> Integer {
    if value? {
      return value!;
    } else {
      return simulate_gamma_poisson(λ!.k, a*λ!.θ);
    }
  }
  
  function observe(x:Integer) -> Real {
    return observe_gamma_poisson(x, λ!.k, a*λ!.θ);
  }

  function update(x:Integer) {
    (λ!.k, λ!.θ) <- update_scaled_gamma_poisson(x, a, λ!.k, λ!.θ);
  }

  function downdate(x:Integer) {
    (λ!.k, λ!.θ) <- downdate_scaled_gamma_poisson(x, a, λ!.k, λ!.θ);
  }

  function pmf(x:Integer) -> Real {
    return pmf_gamma_poisson(x, λ!.k, a*λ!.θ);
  }

  function cdf(x:Integer) -> Real {
    return cdf_gamma_poisson(x, λ!.k, a*λ!.θ);
  }

  function lower() -> Integer? {
    return 0;
  }
}

function DelayScaledGammaPoisson(x:Random<Integer>&, a:Real, λ:DelayGamma) -> 
    DelayScaledGammaPoisson {
  assert a > 0;
  m:DelayScaledGammaPoisson(x, a, λ);
  λ.setChild(m);
  return m;
}
