/*
 * Delayed multivariate linear-Gaussian-Gaussian random variate.
 */
final class DelayLinearMultivariateGaussianMultivariateGaussian(future:Real[_]?,
    futureUpdate:Boolean, A:Real[_,_], m:DelayMultivariateGaussian, c:Real[_],
    S:Real[_,_]) < DelayMultivariateGaussian(future, futureUpdate, A*m.μ + c,
    A*m.Σ*transpose(A) + S) {
  /**
   * Scale.
   */
  A:Real[_,_] <- A;
    
  /**
   * Mean.
   */
  m:DelayMultivariateGaussian& <- m;

  /**
   * Offset.
   */
  c:Real[_] <- c;
  
  /**
   * Likelihood covariance.
   */
  S:Real[_,_] <- S;

  function update(x:Real[_]) {
    (m!.μ, m!.Σ) <- update_linear_multivariate_gaussian_multivariate_gaussian(
        x, A, m!.μ, m!.Σ, c, S);
  }

  function downdate(x:Real[_]) {
    (m!.μ, m!.Σ) <- downdate_linear_multivariate_gaussian_multivariate_gaussian(
        x, A, m!.μ, m!.Σ, c, S);
  }

  function write(buffer:Buffer) {
    buffer.set(value());
  }
}

function DelayLinearMultivariateGaussianMultivariateGaussian(future:Real[_]?,
    futureUpdate:Boolean, A:Real[_,_], μ:DelayMultivariateGaussian, c:Real[_],
    Σ:Real[_,_]) -> DelayLinearMultivariateGaussianMultivariateGaussian {
  m:DelayLinearMultivariateGaussianMultivariateGaussian(future, futureUpdate,
      A, μ, c, Σ);
  μ.setChild(m);
  return m;
}
