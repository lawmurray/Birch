/*
 * Delayed Inverse-gamma gamma random variate.
 */
final class DelayInverseGammaGamma(future:Real?, futureUpdate:Boolean,
    k:Real, θ:DelayInverseGamma) < DelayValue<Real>(future, futureUpdate) {
  /**
   * Shape.
   */
  k:Real <- k;

  /**
   * Scale.
   */
  θ:DelayInverseGamma& <- θ;

  function simulate() -> Real {
    return simulate_compound_gamma(k, θ.α, θ.β);
  }

  function logpdf(x:Real) -> Real {
    return logpdf_compound_gamma(x, k, θ.α, θ.β);
  }

  function update(x:Real) {
    (θ.α, θ.β) <- update_inverse_gamma_gamma(x, k, θ.α, θ.β);
  }

  function downdate(x:Real) {
    (θ.α, θ.β) <- downdate_inverse_gamma_gamma(x, k, θ.α, θ.β);
  }

  function cdf(x:Real) -> Real {
    return cdf_compound_gamma(x, k, θ.α, θ.β);
  }

  function lower() -> Real? {
    return 0.0;
  }

  function write(buffer:Buffer) {
    buffer.set(value());
  }
}

function DelayInverseGammaGamma(future:Real?, futureUpdate:Boolean, k:Real,
    θ:DelayInverseGamma) -> DelayInverseGammaGamma {
  m:DelayInverseGammaGamma(future, futureUpdate, k, θ);
  θ.setChild(m);
  return m;
}
