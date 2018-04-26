/**
 * Synonym for MultivariateIndependentGaussian.
 */
class MultivariateIndependentNormal = MultivariateIndependentGaussian;

/**
 * Create multivariate Gaussian distribution.
 */
function Normal(μ:Expression<Real[_]>, σ2:Expression<Real>) -> MultivariateIndependentNormal {
  return Gaussian(μ, σ2);
}

/**
 * Create multivariate Gaussian distribution.
 */
function Normal(μ:Expression<Real[_]>, σ2:Real) -> MultivariateIndependentNormal {
  return Gaussian(μ, σ2);
}

/**
 * Create multivariate Gaussian distribution.
 */
function Normal(μ:Real[_], σ2:Expression<Real>) -> MultivariateIndependentNormal {
  return Gaussian(μ, σ2);
}

/**
 * Create multivariate Gaussian distribution.
 */
function Normal(μ:Real[_], σ2:Real) -> MultivariateIndependentNormal {
  return Gaussian(μ, σ2);
}
