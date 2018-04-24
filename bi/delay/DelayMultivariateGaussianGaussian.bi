/**
 * Multivariate Gaussian-Gaussian random variable with delayed sampling.
 */
class DelayMultivariateGaussianGaussian(x:Random<Real[_]>,
    μ_0:DelayMultivariateGaussian, Σ:Real[_,_]) <
    DelayMultivariateGaussian(x, μ_0.μ, μ_0.Σ + Σ) {
  /**
   * Prior mean.
   */
  μ_0:DelayMultivariateGaussian <- μ_0;

  /**
   * Marginal mean.
   */
  μ_m:Real[_] <- μ;

  /**
   * Marginal variance.
   */
  Σ_m:Real[_,_] <- Σ;

  function doCondition(x:Real[_]) {
    (μ_0.μ, μ_0.Σ) <- update_multivariate_gaussian_gaussian(x, μ_0.μ, μ_0.Σ, μ_m, Σ_m);
  }
}
