/*
 * Delayed multivariate Gaussian-Gaussian random variate.
 */
class DelayMultivariateGaussianGaussian(x:Random<Real[_]>,
    μ:DelayMultivariateGaussian, Σ:Real[_,_]) <
    DelayMultivariateGaussian(x, μ.μ, μ.Σ + Σ) {
  /**
   * Prior mean.
   */
  μ:DelayMultivariateGaussian <- μ;

  /**
   * Prior covariance.
   */
  Σ:Real[_,_] <- Σ;

  function doCondition(x:Real[_]) {
    (μ.μ, μ.Σ) <- update_multivariate_gaussian_gaussian(x, μ.μ, μ.Σ, Σ);
  }
}

function DelayMultivariateGaussianGaussian(x:Random<Real[_]>,
    μ:DelayMultivariateGaussian, Σ:Real[_,_]) ->
    DelayMultivariateGaussianGaussian {
  m:DelayMultivariateGaussianGaussian(x, μ, Σ);
  return m;
}
