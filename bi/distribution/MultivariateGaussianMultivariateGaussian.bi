/**
 * Multivariate Gaussian-Gaussian distribution.
 */
final class MultivariateGaussianMultivariateGaussian(m:MultivariateGaussian,
    S:Expression<Real[_,_]>) < MultivariateGaussian(m.μ, m.Σ + S) {
  /**
   * Mean.
   */
  m:MultivariateGaussian& <- m;

  /**
   * Likelihood covariance.
   */
  S:Expression<Real[_,_]> <- S;

  function update(x:Real[_]) {
    (m.μ, m.Σ) <- box(update_multivariate_gaussian_multivariate_gaussian(x, m.μ.value(), m.Σ.value(), S.value()));
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
    Σ:Expression<Real[_,_]>) -> MultivariateGaussianMultivariateGaussian {
  m:MultivariateGaussianMultivariateGaussian(μ, Σ);
  m.link();
  return m;
}
