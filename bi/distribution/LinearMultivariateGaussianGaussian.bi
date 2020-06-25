/**
 * Multivariate linear-Gaussian-Gaussian distribution.
 */
final class LinearMultivariateGaussianGaussian(a:Expression<Real[_]>,
    m:MultivariateGaussian, c:Expression<Real>, s2:Expression<Real>) <
    Gaussian(dot(a, m.μ) + c, dot(a, m.Σ*a) + s2) {
  /**
   * Scale.
   */
  a:Expression<Real[_]> <- a;
    
  /**
   * Mean.
   */
  m:MultivariateGaussian& <- m;

  /**
   * Offset.
   */
  c:Expression<Real> <- c;
  
  /**
   * Likelihood covariance.
   */
  s2:Expression<Real> <- s2;

  function update(x:Real) {
    auto m <- this.m;
    (m.μ, m.Σ) <- box(update_linear_multivariate_gaussian_gaussian(
        x, a.value(), m.μ.value(), m.Σ.value(), c.value(), s2.value()));
  }

  function updateLazy(x:Expression<Real>) {
    auto m <- this.m;
    (m.μ, m.Σ) <- update_lazy_linear_multivariate_gaussian_gaussian(
        x, a, m.μ, m.Σ, c, s2);
  }

  function downdate(x:Real) {
    auto m <- this.m;
    (m.μ, m.Σ) <- box(downdate_linear_multivariate_gaussian_gaussian(
        x, a.value(), m.μ.value(), m.Σ.value(), c.value(), s2.value()));
  }

  function link() {
    auto m <- this.m;
    m.setChild(this);
  }
  
  function unlink() {
    auto m <- this.m;
    m.releaseChild(this);
  }
}

function LinearMultivariateGaussianGaussian(a:Expression<Real[_]>,
    μ:MultivariateGaussian, c:Expression<Real>, σ2:Expression<Real>) ->
    LinearMultivariateGaussianGaussian {
  m:LinearMultivariateGaussianGaussian(a, μ, c, σ2);
  m.link();
  return m;
}
