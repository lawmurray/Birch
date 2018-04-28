/**
 * Multivariate Gaussian-Gaussian random variable with delayed sampling.
 */
class DelayMultivariateGaussianGaussian(x:Random<Real[_]>,
    μ:DelayMultivariateGaussian, Σ:Boxed<Real[_,_]>) <
    DelayMultivariateGaussian(x, μ.μ, μ.Σ + Σ.value()) {
  /**
   * Prior mean.
   */
  μ:DelayMultivariateGaussian <- μ;

  /**
   * Prior covariance.
   */
  Σ:Boxed<Real[_,_]> <- Σ;

  function doCondition(x:Real[_]) {
    (μ.μ, μ.Σ) <- update_multivariate_gaussian_gaussian(x, μ.μ, μ.Σ,
        Σ.value());
  }
}
