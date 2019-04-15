/**
 * Synonym for LogGaussian.
 */
final class LogNormal = LogGaussian;

/**
 * Create log-Gaussian distribution.
 */
function LogNormal(μ:Expression<Real>, σ2:Expression<Real>) -> LogNormal {
  return LogGaussian(μ, σ2);
}

/**
 * Create log-Gaussian distribution.
 */
function LogNormal(μ:Expression<Real>, σ2:Real) -> LogNormal {
  return LogGaussian(μ, σ2);
}

/**
 * Create log-Gaussian distribution.
 */
function LogNormal(μ:Real, σ2:Expression<Real>) -> LogNormal {
  return LogGaussian(μ, σ2);
}

/**
 * Create log-Gaussian distribution.
 */
function LogNormal(μ:Real, σ2:Real) -> LogNormal {
  return LogGaussian(μ, σ2);
}
