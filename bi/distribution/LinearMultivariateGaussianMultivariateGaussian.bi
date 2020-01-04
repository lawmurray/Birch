/*
 * ed multivariate linear-Gaussian-Gaussian random variate.
 */
final class LinearMultivariateGaussianMultivariateGaussian(future:Real[_]?,
    futureUpdate:Boolean, A:Real[_,_], m:MultivariateGaussian, c:Real[_],
    S:Real[_,_]) < MultivariateGaussian(future, futureUpdate, A*m.μ + c,
    A*m.Σ*transpose(A) + S) {
  /**
   * Scale.
   */
  A:Real[_,_] <- A;
    
  /**
   * Mean.
   */
  m:MultivariateGaussian& <- m;

  /**
   * Offset.
   */
  c:Real[_] <- c;
  
  /**
   * Likelihood covariance.
   */
  S:Real[_,_] <- S;

  function update(x:Real[_]) {
    (m.μ, m.Σ) <- update_linear_multivariate_gaussian_multivariate_gaussian(
        x, A, m.μ, m.Σ, c, S);
  }

  function downdate(x:Real[_]) {
    (m.μ, m.Σ) <- downdate_linear_multivariate_gaussian_multivariate_gaussian(
        x, A, m.μ, m.Σ, c, S);
  }
}

function LinearMultivariateGaussianMultivariateGaussian(future:Real[_]?,
    futureUpdate:Boolean, A:Real[_,_], μ:MultivariateGaussian, c:Real[_],
    Σ:Real[_,_]) -> LinearMultivariateGaussianMultivariateGaussian {
  m:LinearMultivariateGaussianMultivariateGaussian(future, futureUpdate,
      A, μ, c, Σ);
  μ.setChild(m);
  return m;
}
