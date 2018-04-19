/**
 * Uniform distribution.
 */
class Uniform<Type1,Type2>(l:Type1, u:Type2) < Random<Real> {
  /**
   * Lower bound.
   */
  l:Type1 <- l;
  
  /**
   * Upper bound.
   */
  u:Type2 <- u;

  function update(l:Type1, u:Type2) {
    this.l <- l;
    this.u <- u;
  }

  function doSimulate() -> Real {
    return simulate_uniform(global.value(l), global.value(u));
  }
  
  function doObserve(x:Real) -> Real {
    return observe_uniform(x, global.value(l), global.value(u));
  }
}

/**
 * Create a uniform distribution.
 */
function Uniform(l:Real, u:Real) -> Uniform<Real,Real> {
  m:Uniform<Real,Real>(l, u);
  m.initialize();
  return m;
}

/**
 * Create a uniform distribution.
 */
function Uniform(l:Expression<Real>, u:Real) ->
    Uniform<Expression<Real>,Real> {
  m:Uniform<Expression<Real>,Real>(l, u);
  m.initialize();
  return m;
}

/**
 * Create a uniform distribution.
 */
function Uniform(l:Real, u:Expression<Real>) ->
    Uniform<Real,Expression<Real>> {
  m:Uniform<Real,Expression<Real>>(l, u);
  m.initialize();
  return m;
}

/**
 * Create a uniform distribution.
 */
function Uniform(l:Expression<Real>, u:Expression<Real>) ->
    Uniform<Expression<Real>,Expression<Real>> {
  m:Uniform<Expression<Real>,Expression<Real>>(l, u);
  m.initialize();
  return m;
}
