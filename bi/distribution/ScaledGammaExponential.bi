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
  λ:Gamma <- λ;

  function simulate() -> Real {
    return simulate_lomax(1.0/(a.value()*λ.θ.value()), λ.k.value());
  }

  function logpdf(x:Real) -> Real {
    return logpdf_lomax(x, 1.0/(a.value()*λ.θ.value()), λ.k.value());
  }

  function update(x:Real) {
    (λ.k, λ.θ) <- box(update_scaled_gamma_exponential(x, a.value(),
        λ.k.value(), λ.θ.value()));
  }
  
  function downdate(x:Real) {
    (λ.k, λ.θ) <- box(downdate_scaled_gamma_exponential(x, a.value(),
        λ.k.value(), λ.θ.value()));
  }

  function cdf(x:Real) -> Real? {
    return cdf_lomax(x, 1.0/(a.value()*λ.θ.value()), λ.k.value());
  }

  function quantile(P:Real) -> Real? {
    return quantile_lomax(P, 1.0/(a.value()*λ.θ.value()), λ.k.value());
  }

  function lower() -> Real? {
    return 0.0;
  }

  function link() {
    λ.setChild(this);
  }
  
  function unlink() {
    λ.releaseChild(this);
  }
}

function ScaledGammaExponential(a:Expression<Real>, λ:Gamma) ->
    ScaledGammaExponential {
  m:ScaledGammaExponential(a, λ);
  m.link();
  return m;
}
