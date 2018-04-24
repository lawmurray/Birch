/**
 * Multivariate affine-Gaussian-Gaussian random variable with delayed
 * sampling.
 */
class DelayMultivariateAffineGaussianGaussian(x:Random<Real[_]>, A:Real[_,_],
    μ_0:DelayMultivariateGaussian, c:Real[_], Σ:Real[_,_]) <
    DelayMultivariateGaussian(x, A*μ_0.μ + c, A*μ_0.Σ*trans(A) + Σ) {
  /**
   * Scale.
   */
  A:Real[_,_] <- A;
    
  /**
   * Prior mean.
   */
  μ_0:DelayMultivariateGaussian <- μ_0;

  /**
   * Offset.
   */
  c:Real[_] <- c;

  /**
   * Marginal mean.
   */
  μ_m:Real[_] <- μ;

  /**
   * Marginal variance.
   */
  Σ_m:Real[_,_] <- Σ;

  function doCondition(x:Real[_]) {
    (μ_0.μ, μ_0.Σ) <- update_multivariate_affine_gaussian_gaussian(x, A, μ_0.μ, μ_0.Σ, μ_m, Σ_m);
  }
}
