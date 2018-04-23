/**
 * Uniform distribution.
 */
class Uniform(l:Expression<Real>, u:Expression<Real>) < Random<Real> {
  /**
   * Lower bound.
   */
  l:Expression<Real> <- l;
  
  /**
   * Upper bound.
   */
  u:Expression<Real> <- u;

  function doSimulate() -> Real {
    return simulate_uniform(l.value(), u.value());
  }
  
  function doObserve(x:Real) -> Real {
    return observe_uniform(x, l.value(), u.value());
  }
}

/**
 * Create a uniform distribution.
 */
function Uniform(l:Expression<Real>, u:Expression<Real>) -> Uniform {
  m:Uniform(l, u);
  m.initialize();
  return m;
}

/**
 * Create a uniform distribution.
 */
function Uniform(l:Expression<Real>, u:Real) -> Uniform {
  return Uniform(l, Literal(u));
}

/**
 * Create a uniform distribution.
 */
function Uniform(l:Real, u:Expression<Real>) -> Uniform {
  return Uniform(Literal(l), u);
}

/**
 * Create a uniform distribution.
 */
function Uniform(l:Real, u:Real) -> Uniform {
  return Uniform(Literal(l), Literal(u));
}
