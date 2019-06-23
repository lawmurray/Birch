/*
 * Delayed gamma-Poisson random variate.
 */
final class DelayGammaPoisson(future:Integer?, futureUpdate:Boolean,
    λ:DelayGamma) < DelayDiscrete(future, futureUpdate) {
  /**
   * Rate.
   */
  λ:DelayGamma& <- λ;

  function simulate() -> Integer {
    if value? {
      return value!;
    } else {
      return simulate_gamma_poisson(λ!.k, λ!.θ);
    }
  }
  
  function logpdf(x:Integer) -> Real {
    return logpdf_gamma_poisson(x, λ!.k, λ!.θ);
  }

  function update(x:Integer) {
    (λ!.k, λ!.θ) <- update_gamma_poisson(x, λ!.k, λ!.θ);
  }

  function downdate(x:Integer) {
    (λ!.k, λ!.θ) <- downdate_gamma_poisson(x, λ!.k, λ!.θ);
  }

  function pdf(x:Integer) -> Real {
    return pdf_gamma_poisson(x, λ!.k, λ!.θ);
  }

  function cdf(x:Integer) -> Real {
    return cdf_gamma_poisson(x, λ!.k, λ!.θ);
  }

  function lower() -> Integer? {
    return 0;
  }
}

function DelayGammaPoisson(future:Integer?, futureUpdate:Boolean,
    λ:DelayGamma) ->  DelayGammaPoisson {
  m:DelayGammaPoisson(future, futureUpdate, λ);
  λ.setChild(m);
  return m;
}
