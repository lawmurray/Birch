/*
 * Delayed multivariate Gaussian-Gaussian random variate.
 */
final class DelayMultivariateGaussianGaussian(x:Random<Real[_]>&,
    m:DelayMultivariateGaussian, S:Real[_,_]) <
    DelayMultivariateGaussian(x, m.μ, m.Σ + S) {
  /**
   * Mean.
   */
  m:DelayMultivariateGaussian& <- m;

  /**
   * Likelihood covariance.
   */
  S:Real[_,_] <- S;

  function update(x:Real[_]) {
    (m!.μ, m!.Σ) <- update_multivariate_gaussian_gaussian(x, m!.μ, m!.Σ, S);
  }

  function downdate(x:Real[_]) {
    (m!.μ, m!.Σ) <- downdate_multivariate_gaussian_gaussian(x, m!.μ, m!.Σ, S);
  }
}

function DelayMultivariateGaussianGaussian(x:Random<Real[_]>&,
    μ:DelayMultivariateGaussian, Σ:Real[_,_]) ->
    DelayMultivariateGaussianGaussian {
  m:DelayMultivariateGaussianGaussian(x, μ, Σ);
  μ.setChild(m);
  return m;
}
