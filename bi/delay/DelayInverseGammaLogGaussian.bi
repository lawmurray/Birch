/**
 * Normal-inverse-gamma-log-Gaussian random variable with delayed sampling.
 */
class DelayInverseGammaLogGaussian(x:Random<Real>, μ:Real,
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
    return exp(simulate_inverse_gamma_gaussian(μ, σ2.α, σ2.β));
  }
  
  function doObserve(x:Real) -> Real {
    return observe_inverse_gamma_gaussian(log(x), μ, σ2.α, σ2.β) - log(x);
  }

  function doCondition(x:Real) {
    (σ2.α, σ2.β) <- update_inverse_gamma_gaussian(log(x), μ, σ2.α, σ2.β);
  }
}

function DelayInverseGammaLogGaussian(x:Random<Real>, μ:Real,
    σ2:DelayInverseGamma) -> DelayInverseGammaLogGaussian {
  m:DelayInverseGammaLogGaussian(x, μ, σ2);
  return m;
}
