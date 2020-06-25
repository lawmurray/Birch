/**
 * Scaled gamma-Poisson distribution.
 */
final class ScaledGammaPoisson(a:Expression<Real>, λ:Gamma) < Discrete {
  /**
   * Scale.
   */
  a:Expression<Real> <- a;

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
      return simulate_gamma_poisson(λ.k.value(), a.value()*λ.θ.value());
    }
  }

  function simulateLazy() -> Integer? {
    if value? {
      return value!;
    } else {
      auto λ <- this.λ;
      return simulate_gamma_poisson(λ.k.get(), a.get()*λ.θ.get());
    }
  }
  
  function logpdf(x:Integer) -> Real {
    auto λ <- this.λ;
    return logpdf_gamma_poisson(x, λ.k.value(), a.value()*λ.θ.value());
  }

  function logpdfLazy(x:Expression<Integer>) -> Expression<Real>? {
    auto λ <- this.λ;
    return logpdf_lazy_gamma_poisson(x, λ.k, a*λ.θ);
  }

  function update(x:Integer) {
    auto λ <- this.λ;
    (λ.k, λ.θ) <- box(update_scaled_gamma_poisson(x, a.value(), λ.k.value(),
        λ.θ.value()));
  }

  function updateLazy(x:Expression<Integer>) {
    auto λ <- this.λ;
    (λ.k, λ.θ) <- update_lazy_scaled_gamma_poisson(x, a, λ.k, λ.θ);
  }

  function downdate(x:Integer) {
    auto λ <- this.λ;
    (λ.k, λ.θ) <- box(downdate_scaled_gamma_poisson(x, a.value(), λ.k.value(),
        λ.θ.value()));
  }

  function cdf(x:Integer) -> Real? {
    auto λ <- this.λ;
    return cdf_gamma_poisson(x, λ.k.value(), a.value()*λ.θ.value());
  }

  function quantile(P:Real) -> Integer? {
    auto λ <- this.λ;
    return quantile_gamma_poisson(P, λ.k.value(), a.value()*λ.θ.value());
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

function ScaledGammaPoisson(a:Expression<Real>, λ:Gamma) ->
    ScaledGammaPoisson {
  m:ScaledGammaPoisson(a, λ);
  m.link();
  return m;
}
