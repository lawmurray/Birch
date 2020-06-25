/**
 * Multivariate Gaussian-Gaussian distribution.
 */
final class MultivariateGaussianMultivariateGaussian(m:MultivariateGaussian,
    S:Expression<LLT>) < MultivariateGaussian(m.μ, llt(m.Σ + S)) {
  /**
   * Mean.
   */
  m:MultivariateGaussian& <- m;

  /**
   * Likelihood covariance.
   */
  S:Expression<LLT> <- S;

  function update(x:Real[_]) {
    auto m <- this.m;
    (m.μ, m.Σ) <- box(update_multivariate_gaussian_multivariate_gaussian(x, m.μ.value(), m.Σ.value(), S.value()));
  }

  function updateLazy(x:Expression<Real[_]>) {
    auto m <- this.m;
    (m.μ, m.Σ) <- update_lazy_multivariate_gaussian_multivariate_gaussian(x, m.μ, m.Σ, S);
  }

  function downdate(x:Real[_]) {
    auto m <- this.m;
    (m.μ, m.Σ) <- box(downdate_multivariate_gaussian_multivariate_gaussian(x, m.μ.value(), m.Σ.value(), S.value()));
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

function MultivariateGaussianMultivariateGaussian(μ:MultivariateGaussian,
    Σ:Expression<LLT>) -> MultivariateGaussianMultivariateGaussian {
  m:MultivariateGaussianMultivariateGaussian(μ, Σ);
  m.link();
  return m;
}
