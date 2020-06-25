/**
 * Gamma-Poisson distribution.
 */
final class GammaPoisson(λ:Gamma) < Discrete {
  /**
   * Rate.
   */
  λ:Gamma& <- λ;

  function supportsLazy() -> Boolean {
    return true;
  }

  function simulate() -> Integer {
    if value? {
      return value!;
    } else {
      auto λ <- this.λ;
      return simulate_gamma_poisson(λ.k.value(), λ.θ.value());
    }
  }

  function simulateLazy() -> Integer? {
    if value? {
      return value!;
    } else {
      auto λ <- this.λ;
      return simulate_gamma_poisson(λ.k.get(), λ.θ.get());
    }
  }
  
  function logpdf(x:Integer) -> Real {
    auto λ <- this.λ;
    return logpdf_gamma_poisson(x, λ.k.value(), λ.θ.value());
  }

  function logpdfLazy(x:Expression<Integer>) -> Expression<Real>? {
    auto λ <- this.λ;
    return logpdf_lazy_gamma_poisson(x, λ.k, λ.θ);
  }

  function update(x:Integer) {
    auto λ <- this.λ;
    (λ.k, λ.θ) <- box(update_gamma_poisson(x, λ.k.value(), λ.θ.value()));
  }

  function updateLazy(x:Expression<Integer>) {
    auto λ <- this.λ;
    (λ.k, λ.θ) <- update_lazy_gamma_poisson(x, λ.k, λ.θ);
  }

  function downdate(x:Integer) {
    auto λ <- this.λ;
    (λ.k, λ.θ) <- box(downdate_gamma_poisson(x, λ.k.value(), λ.θ.value()));
  }

  function cdf(x:Integer) -> Real? {
    auto λ <- this.λ;
    return cdf_gamma_poisson(x, λ.k.value(), λ.θ.value());
  }

  function quantile(P:Real) -> Integer? {
    auto λ <- this.λ;
    return quantile_gamma_poisson(P, λ.k.value(), λ.θ.value());
  }

  function lower() -> Integer? {
    return 0;
  }

  function link() {
    auto λ <- this.λ;
    λ.setChild(this);
  }
  
  function unlink() {
    auto λ <- this.λ;
    λ.releaseChild(this);
  }
}

function GammaPoisson(λ:Gamma) -> GammaPoisson {
  m:GammaPoisson(λ);
  m.link();
  return m;
}
