/*
 * Delayed multivariate Gaussian-Gaussian random variate.
 */
final class DelayMultivariateGaussianMultivariateGaussian(future:Real[_]?,
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
    (m.μ, m.Σ) <- update_multivariate_gaussian_multivariate_gaussian(x, m.μ, m.Σ, S);
  }

  function downdate(x:Real[_]) {
    (m.μ, m.Σ) <- downdate_multivariate_gaussian_multivariate_gaussian(x, m.μ, m.Σ, S);
  }

  function write(buffer:Buffer) {
    buffer.set(value());
  }
}

function DelayMultivariateGaussianMultivariateGaussian(future:Real[_]?, futureUpdate:Boolean,
    μ:DelayMultivariateGaussian, Σ:Real[_,_]) ->
    DelayMultivariateGaussianMultivariateGaussian {
  m:DelayMultivariateGaussianMultivariateGaussian(future, futureUpdate, μ, Σ);
  μ.setChild(m);
  return m;
}
