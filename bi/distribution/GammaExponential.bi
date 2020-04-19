/*
 * Grafted gamma-exponential distribution.
 */
final class GammaExponential(λ:Gamma) < Distribution<Real> {
  /**
   * Rate.
   */
  λ:Gamma <- λ;

  function simulate() -> Real {
    return simulate_lomax(1.0/λ.θ.value(), λ.k.value());
  }

  function logpdf(x:Real) -> Real {
    return logpdf_lomax(x, 1.0/λ.θ.value(), λ.k.value());
  }

  function update(x:Real) {
    (λ.k, λ.θ) <- update_gamma_exponential(x, λ.k.value(), λ.θ.value());
  }

  function downdate(x:Real) {
    (λ.k, λ.θ) <- downdate_gamma_exponential(x, λ.k.value(), λ.θ.value());
  }

  function cdf(x:Real) -> Real? {
    return cdf_lomax(x, 1.0/λ.θ.value(), λ.k.value());
  }

  function quantile(P:Real) -> Real? {
    return quantile_lomax(P, 1.0/λ.θ.value(), λ.k.value());
  }

  function lower() -> Real? {
    return 0.0;
  }

  function graftFinalize() -> Boolean {
    if !λ.hasValue() {
      link();
      return true;
    } else {
      return false;
    }
  }

  function link() {
    λ.setChild(this);
  }
  
  function unlink() {
    λ.releaseChild();
  }
}

function GammaExponential(λ:Gamma) -> GammaExponential {
  m:GammaExponential(λ);
  return m;
}
