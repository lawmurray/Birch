/*
 * Grafted scaled gamma-Poisson distribution.
 */
final class ScaledGammaPoisson(a:Expression<Real>, λ:Gamma) < Discrete {
  /**
   * Scale.
   */
  a:Expression<Real> <- a;

  /**
   * Rate.
   */
  λ:Gamma <- λ;

  function simulate() -> Integer {
    if value? {
      return value!;
    } else {
      return simulate_gamma_poisson(λ.k.value(), a.value()*λ.θ.value());
    }
  }
  
  function logpdf(x:Integer) -> Real {
    return logpdf_gamma_poisson(x, λ.k.value(), a.value()*λ.θ.value());
  }

  function update(x:Integer) {
    (λ.k, λ.θ) <- update_scaled_gamma_poisson(x, a.value(), λ.k.value(),
        λ.θ.value());
  }

  function downdate(x:Integer) {
    (λ.k, λ.θ) <- downdate_scaled_gamma_poisson(x, a.value(), λ.k.value(),
        λ.θ.value());
  }

  function cdf(x:Integer) -> Real? {
    return cdf_gamma_poisson(x, λ.k.value(), a.value()*λ.θ.value());
  }

  function quantile(P:Real) -> Integer? {
    return quantile_gamma_poisson(P, λ.k.value(), a.value()*λ.θ.value());
  }

  function lower() -> Integer? {
    return 0;
  }

  function link() {
    λ.setChild(this);
  }
  
  function unlink() {
    λ.releaseChild();
  }
}

function ScaledGammaPoisson(a:Expression<Real>, λ:Gamma) ->
    ScaledGammaPoisson {
  m:ScaledGammaPoisson(a, λ);
  m.link();
  return m;
}
