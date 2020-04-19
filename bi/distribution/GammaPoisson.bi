/*
 * Grafted gamma-Poisson distribution.
 */
final class GammaPoisson(λ:Gamma) < Discrete {
  /**
   * Rate.
   */
  λ:Gamma <- λ;

  function simulate() -> Integer {
    if value? {
      return value!;
    } else {
      return simulate_gamma_poisson(λ.k.value(), λ.θ.value());
    }
  }
  
  function logpdf(x:Integer) -> Real {
    return logpdf_gamma_poisson(x, λ.k.value(), λ.θ.value());
  }

  function update(x:Integer) {
    (λ.k, λ.θ) <- update_gamma_poisson(x, λ.k.value(), λ.θ.value());
  }

  function downdate(x:Integer) {
    (λ.k, λ.θ) <- downdate_gamma_poisson(x, λ.k.value(), λ.θ.value());
  }

  function cdf(x:Integer) -> Real? {
    return cdf_gamma_poisson(x, λ.k.value(), λ.θ.value());
  }

  function quantile(P:Real) -> Integer? {
    return quantile_gamma_poisson(P, λ.k.value(), λ.θ.value());
  }

  function lower() -> Integer? {
    return 0;
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

function GammaPoisson(λ:Gamma) -> GammaPoisson {
  m:GammaPoisson(λ);
  return m;
}
