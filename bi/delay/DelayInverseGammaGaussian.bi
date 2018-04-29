/**
 * Normal-inverse-gamma-Gaussian random variable with delayed sampling.
 */
class DelayInverseGammaGaussian(x:Random<Real>, μ:Real,
    σ2:DelayInverseGamma) < DelayValue<Real>(x) {
  /**
   * Mean.
   */
  μ:Real <- μ;

  /**
   * Variance.
   */
  σ2:DelayInverseGamma <- σ2;

  function doSimulate() -> Real {
    return simulate_inverse_gamma_gaussian(μ, σ2.α, σ2.β);
  }
  
  function doObserve(x:Real) -> Real {
    return observe_inverse_gamma_gaussian(x, μ, σ2.α, σ2.β);
  }

  function doCondition(x:Real) {
    (σ2.α, σ2.β) <- update_inverse_gamma_gaussian(x, μ, σ2.α, σ2.β);
  }
}

function DelayInverseGammaGaussian(x:Random<Real>, μ:Real,
    σ2:DelayInverseGamma) -> DelayInverseGammaGaussian {
  m:DelayInverseGammaGaussian(x, μ, σ2);
  return m;
}
