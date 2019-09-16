/**
 * Multivariate uniform distribution on an orthogonal hyperrectangle.
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

  function valueForward() -> Real[_] {
    assert !delay?;
    return simulate_multivariate_uniform(l, u);
  }

  function observeForward(x:Real[_]) -> Real {
    assert !delay?;
    return logpdf_multivariate_uniform(x, l, u);
  }

  function graft(force:Boolean) {
    if delay? {
      delay!.prune();
    } else if force {
      delay <- DelayIndependentUniform(future, futureUpdate, l, u);
    }
  }
}

/**
 * Create multivariate uniform distribution.
 */
function Uniform(l:Expression<Real[_]>, u:Expression<Real[_]>) -> IndependentUniform {
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
