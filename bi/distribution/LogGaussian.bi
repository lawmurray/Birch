/**
 * Log-Gaussian distribution.
 */
class LogGaussian<Type1,Type2>(μ:Type1, σ2:Type2) < Random<Real> {
  /**
   * Mean after log transformation.
   */
  μ:Type1 <- μ;
  
  /**
   * Variance after log transformation.
   */
  σ2:Type2 <- σ2;

  function update(μ:Type1, σ2:Type2) {
    this.μ <- μ;
    this.σ2 <- σ2;
  }

  function doSimulate() -> Real {
    return simulate_log_gaussian(global.value(μ), global.value(σ2));
  }
  
  function doObserve(x:Real) -> Real {
    return observe_log_gaussian(value(), global.value(μ), global.value(σ2));
  }
}

/**
 * Create log-Gaussian distribution.
 */
function LogGaussian(μ:Real, σ2:Real) -> LogGaussian<Real,Real> {
  m:LogGaussian<Real,Real>(μ, σ2);
  m.initialize();
  return m;
}

/**
 * Create log-Gaussian distribution.
 */
function LogGaussian(μ:Expression<Real>, σ2:Real) ->
    LogGaussian<Expression<Real>,Real> {
  m:LogGaussian<Expression<Real>,Real>(μ, σ2);
  m.initialize();
  return m;
}

/**
 * Create log-Gaussian distribution.
 */
function LogGaussian(μ:Real, σ2:Expression<Real>) ->
    LogGaussian<Real,Expression<Real>> {
  m:LogGaussian<Real,Expression<Real>>(μ, σ2);
  m.initialize();
  return m;
}

/**
 * Create log-Gaussian distribution.
 */
function LogGaussian(μ:Expression<Real>, σ2:Expression<Real>) ->
    LogGaussian<Expression<Real>,Expression<Real>> {
  m:LogGaussian<Expression<Real>,Expression<Real>>(μ, σ2);
  m.initialize();
  return m;
}
