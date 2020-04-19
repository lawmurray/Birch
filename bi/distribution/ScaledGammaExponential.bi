/*
 * Grafted scaled gamma-exponential distribution.
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
    (λ.k, λ.θ) <- update_scaled_gamma_exponential(x, a.value(),
        λ.k.value(), λ.θ.value());
  }
  
  function downdate(x:Real) {
    (λ.k, λ.θ) <- downdate_scaled_gamma_exponential(x, a.value(),
        λ.k.value(), λ.θ.value());
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

  function graftFinalize() -> Boolean {
    a.value();
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

function ScaledGammaExponential(a:Expression<Real>, λ:Gamma) ->
    ScaledGammaExponential {
  m:ScaledGammaExponential(a, λ);
  return m;
}
