/**
 * Multivariate Gaussian-Gaussian distribution.
 */
final class MultivariateGaussianMultivariateGaussian(m:MultivariateGaussian,
    S:Expression<LLT>) < MultivariateGaussian(m.μ,
    llt(matrix(m.Σ) + matrix(S))) {
  /**
   * Mean.
   */
  m:MultivariateGaussian <- m;

  /**
   * Likelihood covariance.
   */
  S:Expression<LLT> <- S;

  function update(x:Real[_]) {
    (m.μ, m.Σ) <- box(update_multivariate_gaussian_multivariate_gaussian(x, m.μ.value(), m.Σ.value(), S.value()));
  }

  function updateLazy(x:Expression<Real[_]>) {
    (m.μ, m.Σ) <- update_lazy_multivariate_gaussian_multivariate_gaussian(x, m.μ, m.Σ, S);
  }

  function downdate(x:Real[_]) {
    (m.μ, m.Σ) <- box(downdate_multivariate_gaussian_multivariate_gaussian(x, m.μ.value(), m.Σ.value(), S.value()));
  }

  function link() {
    m.setChild(this);
  }
  
  function unlink() {
    m.releaseChild(this);
  }
}

function MultivariateGaussianMultivariateGaussian(μ:MultivariateGaussian,
    Σ:Expression<LLT>) -> MultivariateGaussianMultivariateGaussian {
  m:MultivariateGaussianMultivariateGaussian(μ, Σ);
  m.link();
  return m;
}
