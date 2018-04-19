/**
 * Gaussian distribution.
 */
class Gaussian<Type1,Type2>(μ:Type1, σ2:Type2) < Random<Real> {
  /**
   * Mean.
   */
  μ:Type1 <- μ;
  
  /**
   * Variance.
   */
  σ2:Type2 <- σ2;

  function update(μ:Type1, σ2:Type2) {
    this.μ <- μ;
    this.σ2 <- σ2;
  }

  function doSimulate() -> Real {
    return simulate_gaussian(global.value(μ), global.value(σ2));
  }
  
  function doObserve(x:Real) -> Real {
    return observe_gaussian(x, global.value(μ), global.value(σ2));
  }
}

/**
 * Create Gaussian distribution.
 */
function Gaussian(μ:Real, σ2:Real) -> Gaussian<Real,Real> {
  m:Gaussian<Real,Real>(μ, σ2);
  m.initialize();
  return m;
}

/**
 * Create Gaussian distribution.
 */
function Gaussian(μ:Expression<Real>, σ2:Real) ->
    Gaussian<Expression<Real>,Real> {
  m:Gaussian<Expression<Real>,Real>(μ, σ2);
  m.initialize();
  return m;
}

/**
 * Create Gaussian distribution.
 */
function Gaussian(μ:Real, σ2:Expression<Real>) ->
    Gaussian<Real,Expression<Real>> {
  m:Gaussian<Real,Expression<Real>>(μ, σ2);
  m.initialize();
  return m;
}

/**
 * Create Gaussian distribution.
 */
function Gaussian(μ:Expression<Real>, σ2:Expression<Real>) ->
    Gaussian<Expression<Real>,Expression<Real>> {
  m:Gaussian<Expression<Real>,Expression<Real>>(μ, σ2);
  m.initialize();
  return m;
}
