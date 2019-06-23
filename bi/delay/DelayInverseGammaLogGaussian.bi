/*
 * Delayed normal-inverse-gamma-log-Gaussian random variate.
 */
final class DelayInverseGammaLogGaussian(future:Real?, futureUpdate:Boolean,
    μ:Real, σ2:DelayInverseGamma) < DelayValue<Real>(future, futureUpdate) {
  /**
   * Mean.
   */
  μ:Real <- μ;

  /**
   * Variance.
   */
  σ2:DelayInverseGamma& <- σ2;

  function simulate() -> Real {
    return exp(simulate_inverse_gamma_gaussian(μ, σ2!.α, σ2!.β));
  }
  
  function logpdf(x:Real) -> Real {
    return logpdf_inverse_gamma_gaussian(log(x), μ, σ2!.α, σ2!.β) - log(x);
  }

  function update(x:Real) {
    (σ2!.α, σ2!.β) <- update_inverse_gamma_gaussian(log(x), μ, σ2!.α, σ2!.β);
  }

  function downdate(x:Real) {
    (σ2!.α, σ2!.β) <- downdate_inverse_gamma_gaussian(log(x), μ, σ2!.α, σ2!.β);
  }

  function pdf(x:Real) -> Real {
    return pdf_inverse_gamma_gaussian(log(x), μ, σ2!.α, σ2!.β)/x;
  }

  function cdf(x:Real) -> Real {
    return cdf_inverse_gamma_gaussian(log(x), μ, σ2!.α, σ2!.β);
  }

  function lower() -> Real? {
    return 0.0;
  }

  function write(buffer:Buffer) {
    buffer.set(value());
  }
}

function DelayInverseGammaLogGaussian(future:Real?, futureUpdate:Boolean,
    μ:Real, σ2:DelayInverseGamma) -> DelayInverseGammaLogGaussian {
  m:DelayInverseGammaLogGaussian(future, futureUpdate, μ, σ2);
  σ2.setChild(m);
  return m;
}
