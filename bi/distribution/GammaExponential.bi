/*
 * ed gamma-Exponential random variate.
 */
final class GammaExponential(λ:Gamma) < Distribution<Real> {
  /**
   * Rate.
   */
  λ:Gamma& <- λ;

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

  function quantile(P:Real) -> Real? {
    return quantile_lomax(P, 1.0/λ.θ, λ.k);
  }

  function lower() -> Real? {
    return 0.0;
  }
}

function GammaExponential(λ:Gamma) -> GammaExponential {
  m:GammaExponential(λ);
  λ.setChild(m);
  return m;
}
