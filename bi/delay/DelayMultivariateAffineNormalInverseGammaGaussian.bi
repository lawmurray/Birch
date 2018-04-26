/**
 * Multivariate affine-normal-inverse-gamma-Gaussian random variable with
 * delayed sampling.
 */
class DelayMultivariateAffineNormalInverseGammaGaussian(x:Random<Real[_]>,
    Λ:Real[_,_], μ:DelayMultivariateNormalInverseGamma, c:Real[_]) <
    DelayValue<Real[_]>(x) {
  /**
   * Scale.
   */
  A:Real[_,_];
    
  /**
   * Mean.
   */
  μ:DelayMultivariateNormalInverseGamma <- μ;

  /**
   * Offset.
   */
  c:Real[_] <- c;

  function doSimulate() -> Real[_] {
    return simulate_multivariate_affine_normal_inverse_gamma_gaussian(A, μ.μ,
        c, μ.Λ, μ.σ2.α, μ.σ2.β);
  }
  
  function doObserve(x:Real[_]) -> Real {
    return observe_multivariate_affine_normal_inverse_gamma_gaussian(x, A,
        μ.μ, c, μ.Λ, μ.σ2.α, μ.σ2.β);
  }

  function doCondition(x:Real[_]) {
    (μ.μ, μ.Λ, μ.σ2.α, μ.σ2.β) <- update_multivariate_affine_normal_inverse_gamma_gaussian(
        x, A, μ.μ, c, μ.Λ, μ.σ2.α, μ.σ2.β);
  }
}
