/*
 * Grafted multivariate linear-Gaussian-Gaussian distribution.
 */
final class LinearMultivariateGaussianGaussian(a:Expression<Real[_]>,
    m:MultivariateGaussian, c:Expression<Real>, s2:Expression<Real>) <
    GraftedGaussian(dot(a, m.μ) + c, dot(a, m.Σ*a) + s2) {
  /**
   * Scale.
   */
  a:Expression<Real[_]> <- a;
    
  /**
   * Mean.
   */
  m:MultivariateGaussian <- m;

  /**
   * Offset.
   */
  c:Expression<Real> <- c;
  
  /**
   * Likelihood covariance.
   */
  s2:Expression<Real> <- s2;

  function update(x:Real) {
    (m.μ, m.Σ) <- update_linear_multivariate_gaussian_gaussian(
        x, a.value(), m.μ.value(), m.Σ.value(), c.value(), s2.value());
  }

  function downdate(x:Real) {
    (m.μ, m.Σ) <- downdate_linear_multivariate_gaussian_gaussian(
        x, a.value(), m.μ.value(), m.Σ.value(), c.value(), s2.value());
  }

  function updateLazy(x:Expression<Real>) {
    (m.μ, m.Σ) <- update_lazy_linear_multivariate_gaussian_gaussian(
        x, a, m.μ, m.Σ, c, s2);
  }

  function graftFinalize() -> Boolean {
    a.value();
    c.value();
    s2.value();
    if !m.isRealized() {
      link();
      return true;
    } else {
      return false;
    }
  }

  function link() {
    m.setChild(this);
  }
  
  function unlink() {
    m.releaseChild();
  }
}

function LinearMultivariateGaussianGaussian(a:Expression<Real[_]>,
    μ:MultivariateGaussian, c:Expression<Real>, σ2:Expression<Real>) ->
    LinearMultivariateGaussianGaussian {
  m:LinearMultivariateGaussianGaussian(a, μ, c, σ2);
  return m;
}
