/**
 * Scaled gamma-exponential distribution.
 */
class ScaledGammaExponential(a:Expression<Real>, λ:Gamma) <
    Distribution<Real> {
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

  function simulate() -> Real {
    auto λ <- this.λ;
    return simulate_lomax(1.0/(a.value()*λ.θ.value()), λ.k.value());
  }

  function simulateLazy() -> Real? {
    auto λ <- this.λ;
    return simulate_lomax(1.0/(a.get()*λ.θ.get()), λ.k.get());
  }

  function logpdf(x:Real) -> Real {
    auto λ <- this.λ;
    return logpdf_lomax(x, 1.0/(a.value()*λ.θ.value()), λ.k.value());
  }

  function logpdfLazy(x:Expression<Real>) -> Expression<Real>? {
    auto λ <- this.λ;
    return logpdf_lazy_lomax(x, 1.0/(a*λ.θ), λ.k);
  }

  function update(x:Real) {
    auto λ <- this.λ;
    (λ.k, λ.θ) <- box(update_scaled_gamma_exponential(x, a.value(),
        λ.k.value(), λ.θ.value()));
  }

  function updateLazy(x:Expression<Real>) {
    auto λ <- this.λ;
    (λ.k, λ.θ) <- update_lazy_scaled_gamma_exponential(x, a, λ.k, λ.θ);
  }
  
  function downdate(x:Real) {
    auto λ <- this.λ;
    (λ.k, λ.θ) <- box(downdate_scaled_gamma_exponential(x, a.value(),
        λ.k.value(), λ.θ.value()));
  }

  function cdf(x:Real) -> Real? {
    auto λ <- this.λ;
    return cdf_lomax(x, 1.0/(a.value()*λ.θ.value()), λ.k.value());
  }

  function quantile(P:Real) -> Real? {
    auto λ <- this.λ;
    return quantile_lomax(P, 1.0/(a.value()*λ.θ.value()), λ.k.value());
  }

  function lower() -> Real? {
    return 0.0;
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

function ScaledGammaExponential(a:Expression<Real>, λ:Gamma) ->
    ScaledGammaExponential {
  m:ScaledGammaExponential(a, λ);
  m.link();
  return m;
}
