/**
 * Gamma distribution.
 */
class Gamma<Type1,Type2>(k:Type1, θ:Type2) < Random<Real> {
  /**
   * Shape.
   */
  k:Type1 <- k;
  
  /**
   * Scale.
   */
  θ:Type2 <- θ;

  function update(k:Type1, θ:Type2) {
    this.k <- k;
    this.θ <- θ;
  }

  function doSimulate() -> Real {
    return simulate_gamma(global.value(k), global.value(θ));
  }
  
  function doObserve(x:Real) -> Real {
    return observe_gamma(x, global.value(k), global.value(θ));
  }
}

/**
 * Create gamma distribution.
 */
function Gamma(k:Real, θ:Real) -> Gamma<Real,Real> {
  m:Gamma<Real,Real>(k, θ);
  m.initialize();
  return m;
}

/**
 * Create gamma distribution.
 */
function Gamma(k:Expression<Real>, θ:Real) -> Gamma<Expression<Real>,Real> {
  m:Gamma<Expression<Real>,Real>(k, θ);
  m.initialize();
  return m;
}

/**
 * Create gamma distribution.
 */
function Gamma(k:Real, θ:Expression<Real>) -> Gamma<Real,Expression<Real>> {
  m:Gamma<Real,Expression<Real>>(k, θ);
  m.initialize();
  return m;
}

/**
 * Create gamma distribution.
 */
function Gamma(k:Expression<Real>, θ:Expression<Real>) ->
    Gamma<Expression<Real>,Expression<Real>> {
  m:Gamma<Expression<Real>,Expression<Real>>(k, θ);
  m.initialize();
  return m;
}
