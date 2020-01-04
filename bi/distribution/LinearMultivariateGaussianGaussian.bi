/*
 * ed multivariate linear-Gaussian-Gaussian random variate.
 */
final class LinearMultivariateGaussianGaussian(future:Real?,
    futureUpdate:Boolean, a:Expression<Real[_]>, m:MultivariateGaussian,
    c:Expression<Real>, s2:Expression<Real>) < Gaussian(future,
    futureUpdate, dot(a, m.μ) + c, dot(a, m.Σ*a) + s2) {
  /**
   * Scale.
   */
  auto a <- a;
    
  /**
   * Mean.
   */
  m:MultivariateGaussian& <- m;

  /**
   * Offset.
   */
  auto c <- c;
  
  /**
   * Likelihood covariance.
   */
  auto s2 <- s2;

  function update(x:Real) {
    (m.μ, m.Σ) <- update_linear_multivariate_gaussian_gaussian(
        x, a, m.μ, m.Σ, c, s2);
  }

  function downdate(x:Real) {
    (m.μ, m.Σ) <- downdate_linear_multivariate_gaussian_gaussian(
        x, a, m.μ, m.Σ, c, s2);
  }
}

function LinearMultivariateGaussianGaussian(future:Real?,
    futureUpdate:Boolean, a:Expression<Real[_]>, μ:MultivariateGaussian,
    c:Expression<Real>, σ2:Expression<Real>) ->
    LinearMultivariateGaussianGaussian {
  m:LinearMultivariateGaussianGaussian(future, futureUpdate, a.graft(),
      μ, c.graft(), σ2.graft());
  μ.setChild(m);
  return m;
}
