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

  function doGraft() -> Delay? {
    return DelayUniform(this, l, u);
  }
}

/**
 * Create a uniform distribution.
 */
function Uniform(l:Expression<Real>, u:Expression<Real>) -> Uniform {
  m:Uniform(l, u);
  return m;
}

/**
 * Create a uniform distribution.
 */
function Uniform(l:Expression<Real>, u:Real) -> Uniform {
  return Uniform(l, Boxed(u));
}

/**
 * Create a uniform distribution.
 */
function Uniform(l:Real, u:Expression<Real>) -> Uniform {
  return Uniform(Boxed(l), u);
}

/**
 * Create a uniform distribution.
 */
function Uniform(l:Real, u:Real) -> Uniform {
  return Uniform(Boxed(l), Boxed(u));
}
