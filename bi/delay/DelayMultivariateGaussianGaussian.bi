/*
 * Delayed multivariate Gaussian-Gaussian random variate.
 */
class DelayMultivariateGaussianGaussian(x:Random<Real[_]>&,
    μ_0:DelayMultivariateGaussian, Σ:Real[_,_]) <
    DelayMultivariateGaussian(x, μ_0.μ, μ_0.Σ + Σ) {
  /**
   * Prior mean.
   */
  μ_0:DelayMultivariateGaussian& <- μ_0;

  /**
   * Marginal mean.
   */
  μ_m:Real[_] <- this.μ;

  /**
   * Marginal covariance.
   */
  Σ_m:Real[_,_] <- this.Σ;

  function condition(x:Real[_]) {
    (μ_0!.μ, μ_0!.Σ) <- update_multivariate_gaussian_gaussian(x, μ_0!.μ, μ_0!.Σ, 
        μ_m, Σ_m);
  }
}

function DelayMultivariateGaussianGaussian(x:Random<Real[_]>&,
    μ:DelayMultivariateGaussian, Σ:Real[_,_]) ->
    DelayMultivariateGaussianGaussian {
  m:DelayMultivariateGaussianGaussian(x, μ, Σ);
  μ.setChild(m);
  return m;
}
