/**
 * Multivariate normal-inverse-gamma-Gaussian random variable with delayed
 * sampling.
 */
class DelayMultivariateInverseGammaGaussian(x:Random<Real[_]>, μ:Real[_],
    σ2:DelayInverseGamma) < DelayValue<Real[_]>(x) {
  /**
   * Mean.
   */
  μ:Real[_] <- μ;

  /**
   * Variance.
   */
  σ2:DelayInverseGamma <- σ2;

  function doSimulate() -> Real[_] {
    return simulate_multivariate_inverse_gamma_gaussian(μ, σ2.α, σ2.β);
  }
  
  function doObserve(x:Real[_]) -> Real {
    return observe_multivariate_inverse_gamma_gaussian(x, μ, σ2.α, σ2.β);
  }

  function doCondition(x:Real[_]) {
    (σ2.α, σ2.β) <- update_multivariate_inverse_gamma_gaussian(x, μ, σ2.α, σ2.β);
  }
}

function DelayMultivariateInverseGammaGaussian(x:Random<Real[_]>, μ:Real[_],
    σ2:DelayInverseGamma) -> DelayMultivariateInverseGammaGaussian {
  m:DelayMultivariateInverseGammaGaussian(x, μ, σ2);
  return m;
}
