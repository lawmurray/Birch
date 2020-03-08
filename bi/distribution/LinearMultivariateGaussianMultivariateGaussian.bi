/*
 * ed multivariate linear-Gaussian-Gaussian random variate.
 */
final class LinearMultivariateGaussianMultivariateGaussian(
    A:Expression<Real[_,_]>, m:MultivariateGaussian, c:Expression<Real[_]>,
    S:Expression<Real[_,_]>) < MultivariateGaussian(A*m.μ + c,
    A*m.Σ*transpose(A) + S) {
  /**
   * Scale.
   */
  A:Expression<Real[_,_]> <- A;
    
  /**
   * Mean.
   */
  m:MultivariateGaussian& <- m;

  /**
   * Offset.
   */
  c:Expression<Real[_]> <- c;
  
  /**
   * Likelihood covariance.
   */
  S:Expression<Real[_,_]> <- S;

  function update(x:Real[_]) {
    (m.μ, m.Σ) <- update_linear_multivariate_gaussian_multivariate_gaussian(
        x, A.value(), m.μ.value(), m.Σ.value(), c.value(), S.value());
  }

  function downdate(x:Real[_]) {
    (m.μ, m.Σ) <- downdate_linear_multivariate_gaussian_multivariate_gaussian(
        x, A.value(), m.μ.value(), m.Σ.value(), c.value(), S.value());
  }

  function updateLazy(x:Expression<Real[_]>) {
    (m.μ, m.Σ) <- update_lazy_linear_multivariate_gaussian_multivariate_gaussian(
        x, A, m.μ, m.Σ, c, S);
  }
}

function LinearMultivariateGaussianMultivariateGaussian(
    A:Expression<Real[_,_]>, μ:MultivariateGaussian, c:Expression<Real[_]>,
    Σ:Expression<Real[_,_]>) ->
    LinearMultivariateGaussianMultivariateGaussian {
  m:LinearMultivariateGaussianMultivariateGaussian(A, μ, c, Σ);
  μ.setChild(m);
  return m;
}
