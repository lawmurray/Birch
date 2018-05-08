/*
 * Delayed multivariate Gaussian-Gaussian random variate.
 */
class DelayMultivariateGaussianGaussian(μ:DelayMultivariateGaussian,
    Σ:Real[_,_]) < DelayMultivariateGaussian(μ.μ, μ.Σ + Σ) {
  /**
   * Prior mean.
   */
  μ:DelayMultivariateGaussian <- μ;

  /**
   * Prior covariance.
   */
  Σ:Real[_,_] <- Σ;

  function condition(x:Real[_]) {
    (μ.μ, μ.Σ) <- update_multivariate_gaussian_gaussian(x, μ.μ, μ.Σ, Σ);
  }
}

function DelayMultivariateGaussianGaussian(μ:DelayMultivariateGaussian,
    Σ:Real[_,_]) -> DelayMultivariateGaussianGaussian {
  m:DelayMultivariateGaussianGaussian(μ, Σ);
  return m;
}
