/*
 * Delayed inverse-gamma-Gaussian random variate.
 */
final class DelayInverseGammaGaussian(future:Real?, futureUpdate:Boolean,
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
    return simulate_inverse_gamma_gaussian(μ, σ2!.α, σ2!.β);
  }
  
  function observe(x:Real) -> Real {
    return observe_inverse_gamma_gaussian(x, μ, σ2!.α, σ2!.β);
  }

  function update(x:Real) {
    (σ2!.α, σ2!.β) <- update_inverse_gamma_gaussian(x, μ, σ2!.α, σ2!.β);
  }

  function downdate(x:Real) {
    (σ2!.α, σ2!.β) <- downdate_inverse_gamma_gaussian(x, μ, σ2!.α, σ2!.β);
  }

  function pdf(x:Real) -> Real {
    return pdf_inverse_gamma_gaussian(x, μ, σ2!.α, σ2!.β);
  }

  function cdf(x:Real) -> Real {
    return cdf_inverse_gamma_gaussian(x, μ, σ2!.α, σ2!.β);
  }
}

function DelayInverseGammaGaussian(future:Real?, futureUpdate:Boolean, μ:Real,
    σ2:DelayInverseGamma) -> DelayInverseGammaGaussian {
  m:DelayInverseGammaGaussian(future, futureUpdate, μ, σ2);
  σ2.setChild(m);
  return m;
}
