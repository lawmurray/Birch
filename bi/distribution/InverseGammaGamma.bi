/*
 * ed inverse-gamma-gamma random variate.
 */
final class InverseGammaGamma(future:Real?, futureUpdate:Boolean,
    k:Real, θ:InverseGamma) < Distribution<Real>(future, futureUpdate) {
  /**
   * Shape.
   */
  k:Real <- k;

  /**
   * Scale.
   */
  θ:InverseGamma& <- θ;

  function simulate() -> Real {
    return simulate_inverse_gamma_gamma(k, θ.α, θ.β);
  }

  function logpdf(x:Real) -> Real {
    return logpdf_inverse_gamma_gamma(x, k, θ.α, θ.β);
  }

  function update(x:Real) {
    (θ.α, θ.β) <- update_inverse_gamma_gamma(x, k, θ.α, θ.β);
  }

  function downdate(x:Real) {
    (θ.α, θ.β) <- downdate_inverse_gamma_gamma(x, k, θ.α, θ.β);
  }

  function cdf(x:Real) -> Real? {
    return cdf_inverse_gamma_gamma(x, k, θ.α, θ.β);
  }

  function lower() -> Real? {
    return 0.0;
  }
}

function InverseGammaGamma(future:Real?, futureUpdate:Boolean, k:Real,
    θ:InverseGamma) -> InverseGammaGamma {
  m:InverseGammaGamma(future, futureUpdate, k, θ);
  θ.setChild(m);
  return m;
}
