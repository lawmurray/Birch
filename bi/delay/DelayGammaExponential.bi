/*
 * Delayed gamma-Exponential random variate.
 */
final class DelayGammaExponential(x:Random<Real>&, λ:DelayGamma) <
    DelayValue<Real>(x) {
  /**
   * Rate.
   */
  λ:DelayGamma& <- λ;

  function simulate() -> Real {
    return simulate_lomax(1.0/λ!.θ, λ!.k);
  }

  function observe(x:Real) -> Real {
    return observe_lomax(x, 1.0/λ!.θ, λ!.k);
  }

  function update(x:Real) {
    (λ!.k, λ!.θ) <- update_gamma_exponential(x, λ!.k, λ!.θ);
  }

  function downdate(x:Real) {
    (λ!.k, λ!.θ) <- downdate_gamma_exponential(x, λ!.k, λ!.θ);
  }

  function pdf(x:Real) -> Real {
    return pdf_lomax(x, 1.0/λ!.θ, λ!.k);
  }

  function cdf(x:Real) -> Real {
    return cdf_lomax(x, 1.0/λ!.θ, λ!.k);
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
