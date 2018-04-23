/**
 * Synonym for MultivariateGaussian.
 */
class MultivariateNormal = MultivariateGaussian;

/**
 * Create multivariate Gaussian distribution.
 */
function Normal(μ:Expression<Real[_]>, Σ:Expression<Real[_,_]>) -> MultivariateNormal {
  return Gaussian(μ, Σ);
}

/**
 * Create multivariate Gaussian distribution.
 */
function Normal(μ:Expression<Real[_]>, Σ:Real[_,_]) -> MultivariateNormal {
  return Gaussian(μ, Σ);
}

/**
 * Create multivariate Gaussian distribution.
 */
function Normal(μ:Real[_], Σ:Expression<Real[_,_]>) -> MultivariateNormal {
  return Gaussian(μ, Σ);
}

/**
 * Create multivariate Gaussian distribution.
 */
function Normal(μ:Real[_], Σ:Real[_,_]) -> MultivariateNormal {
  return Gaussian(μ, Σ);
}
