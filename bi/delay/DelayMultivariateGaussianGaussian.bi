/*
 * Delayed multivariate Gaussian-Gaussian random variate.
 */
final class DelayMultivariateGaussianGaussian(future:Real[_]?,
    futureUpdate:Boolean, m:DelayMultivariateGaussian, S:Real[_,_]) <
    DelayMultivariateGaussian(future, futureUpdate, m.μ, m.Σ + S) {
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

function DelayMultivariateGaussianGaussian(future:Real[_]?, futureUpdate:Boolean,
    μ:DelayMultivariateGaussian, Σ:Real[_,_]) ->
    DelayMultivariateGaussianGaussian {
  m:DelayMultivariateGaussianGaussian(future, futureUpdate, μ, Σ);
  μ.setChild(m);
  return m;
}
