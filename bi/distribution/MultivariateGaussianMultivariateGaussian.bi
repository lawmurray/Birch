/*
 * ed multivariate Gaussian-Gaussian random variate.
 */
final class MultivariateGaussianMultivariateGaussian(
    m:MultivariateGaussian, S:Real[_,_]) <
    MultivariateGaussian(m.μ, m.Σ + S) {
  /**
   * Mean.
   */
  m:MultivariateGaussian& <- m;

  /**
   * Likelihood covariance.
   */
  S:Real[_,_] <- S;

  function update(x:Real[_]) {
    (m.μ, m.Σ) <- update_multivariate_gaussian_multivariate_gaussian(x,
        m.μ, m.Σ, S);
  }

  function downdate(x:Real[_]) {
    (m.μ, m.Σ) <- downdate_multivariate_gaussian_multivariate_gaussian(x,
        m.μ, m.Σ, S);
  }
}

function MultivariateGaussianMultivariateGaussian(μ:MultivariateGaussian,
    Σ:Real[_,_]) -> MultivariateGaussianMultivariateGaussian {
  m:MultivariateGaussianMultivariateGaussian(μ, Σ);
  μ.setChild(m);
  return m;
}
