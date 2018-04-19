/**
 * Synonym for LogGaussian.
 */
class LogNormal<Type1,Type2> = LogGaussian<Type1,Type2>;

/**
 * Create log-Gaussian distribution.
 */
function LogNormal(μ:Real, σ2:Real) -> LogGaussian<Real,Real> {
  return LogGaussian(μ, σ2);
}

/**
 * Create log-Gaussian distribution.
 */
function LogNormal(μ:Expression<Real>, σ2:Real) ->
    LogGaussian<Expression<Real>,Real> {
  return LogGaussian(μ, σ2);
}

/**
 * Create log-Gaussian distribution.
 */
function LogNormal(μ:Real, σ2:Expression<Real>) ->
    LogGaussian<Real,Expression<Real>> {
  return LogGaussian(μ, σ2);
}

/**
 * Create log-Gaussian distribution.
 */
function LogNormal(μ:Expression<Real>, σ2:Expression<Real>) ->
    LogGaussian<Expression<Real>,Expression<Real>> {
  return LogGaussian(μ, σ2);
}
