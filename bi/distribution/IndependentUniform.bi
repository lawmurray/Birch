/**
 * Uniform distribution on an orthogonal hyperrectangle.
 */
final class IndependentUniform(l:Expression<Real[_]>, u:Expression<Real[_]>) < Distribution<Real[_]> {
  /**
   * Lower bound.
   */
  l:Expression<Real[_]> <- l;
  
  /**
   * Upper bound.
   */
  u:Expression<Real[_]> <- u;

  function rows() -> Integer {
    return l.rows();
  }

  function graft() {
    if delay? {
      delay!.prune();
    } else {
      delay <- DelayIndependentUniform(future, futureUpdate, l, u);
    }
  }
}

/**
 * Create multivariate uniform distribution.
 */
function Uniform(l:Expression<Real[_]>, u:Expression<Real[_]>) -> IndependentUniform {
  assert l.rows() == u.rows();
  m:IndependentUniform(l, u);
  return m;
}

/**
 * Create multivariate uniform distribution.
 */
function Uniform(l:Expression<Real[_]>, u:Real[_]) -> IndependentUniform {
  return Uniform(l, Boxed(u));
}

/**
 * Create multivariate uniform distribution.
 */
function Uniform(l:Real[_], u:Expression<Real[_]>) -> IndependentUniform {
  return Uniform(Boxed(l), u);
}

/**
 * Create multivariate uniform distribution.
 */
function Uniform(l:Real[_], u:Real[_]) -> IndependentUniform {
  return Uniform(Boxed(l), Boxed(u));
}
