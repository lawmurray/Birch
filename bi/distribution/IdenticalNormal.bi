/**
 * Synonym for IdenticalGaussian.
 */
final class IdenticalNormal = IdenticalGaussian;

/**
 * Create multivariate Gaussian distribution.
 */
function Normal(μ:Expression<Real[_]>, σ2:Expression<Real>) -> IdenticalNormal {
  return Gaussian(μ, σ2);
}

/**
 * Create multivariate Gaussian distribution.
 */
function Normal(μ:Expression<Real[_]>, σ2:Real) -> IdenticalNormal {
  return Gaussian(μ, σ2);
}

/**
 * Create multivariate Gaussian distribution.
 */
function Normal(μ:Real[_], σ2:Expression<Real>) -> IdenticalNormal {
  return Gaussian(μ, σ2);
}

/**
 * Create multivariate Gaussian distribution.
 */
function Normal(μ:Real[_], σ2:Real) -> IdenticalNormal {
  return Gaussian(μ, σ2);
}
