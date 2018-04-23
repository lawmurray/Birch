/**
 * Synonym for Gaussian.
 */
class Normal = Gaussian;

/**
 * Create Gaussian distribution.
 */
function Normal(μ:Expression<Real>, σ2:Expression<Real>) -> Gaussian {
  return Gaussian(μ, σ2);
}

/**
 * Create Gaussian distribution.
 */
function Normal(μ:Expression<Real>, σ2:Real) -> Gaussian {
  return Gaussian(μ, σ2);
}

/**
 * Create Gaussian distribution.
 */
function Normal(μ:Real, σ2:Expression<Real>) -> Gaussian {
  return Gaussian(μ, σ2);
}

/**
 * Create Gaussian distribution.
 */
function Normal(μ:Real, σ2:Real) -> Gaussian {
  return Gaussian(μ, σ2);
}
