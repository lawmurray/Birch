/**
 * Multivariate linear-Gaussian-Gaussian distribution.
 */
final class LinearMultivariateGaussianMultivariateGaussian(
    A:Expression<Real[_,_]>, m:MultivariateGaussian, c:Expression<Real[_]>,
    S:Expression<LLT>) < MultivariateGaussian(A*m.μ + c,
    llt(A*m.Σ*transpose(A) + matrix(S))) {
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
  S:Expression<LLT> <- S;

  function update(x:Real[_]) {
    auto m <- this.m;
    (m.μ, m.Σ) <- box(update_linear_multivariate_gaussian_multivariate_gaussian(
        x, A.value(), m.μ.value(), m.Σ.value(), c.value(), S.value()));
  }

  function updateLazy(x:Expression<Real[_]>) {
    auto m <- this.m;
    (m.μ, m.Σ) <- update_lazy_linear_multivariate_gaussian_multivariate_gaussian(
        x, A, m.μ, m.Σ, c, S);
  }

  function downdate(x:Real[_]) {
    auto m <- this.m;
    (m.μ, m.Σ) <- box(downdate_linear_multivariate_gaussian_multivariate_gaussian(
        x, A.value(), m.μ.value(), m.Σ.value(), c.value(), S.value()));
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

function LinearMultivariateGaussianMultivariateGaussian(
    A:Expression<Real[_,_]>, μ:MultivariateGaussian, c:Expression<Real[_]>,
    Σ:Expression<LLT>) ->
    LinearMultivariateGaussianMultivariateGaussian {
  m:LinearMultivariateGaussianMultivariateGaussian(A, μ, c, Σ);
  m.link();
  return m;
}
