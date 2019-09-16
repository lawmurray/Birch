/**
 * Multivariate uniform distribution over integers on an orthogonal lattice.
 */
final class IndependentUniformInteger(l:Expression<Integer[_]>,
    u:Expression<Integer[_]>) < Distribution<Integer[_]> {
  /**
   * Lower bound.
   */
  l:Expression<Integer[_]> <- l;
  
  /**
   * Upper bound.
   */
  u:Expression<Integer[_]> <- u;

  function valueForward() -> Integer[_] {
    assert !delay?;
    return simulate_multivariate_uniform_int(l, u);
  }

  function observeForward(x:Integer[_]) -> Real {
    assert !delay?;
    return logpdf_multivariate_uniform_int(x, l, u);
  }

  function graft(force:Boolean) {
    if delay? {
      delay!.prune();
    } else if force {
      delay <- DelayIndependentUniformInteger(future, futureUpdate, l, u);
    }
  }
}

/**
 * Create multivariate uniform distribution over integers.
 */
function Uniform(l:Expression<Integer[_]>, u:Expression<Integer[_]>) -> IndependentUniformInteger {
  m:IndependentUniformInteger(l, u);
  return m;
}

/**
 * Create multivariate uniform distribution over integers.
 */
function Uniform(l:Expression<Integer[_]>, u:Integer[_]) -> IndependentUniformInteger {
  return Uniform(l, Boxed(u));
}

/**
 * Create multivariate uniform distribution over integers.
 */
function Uniform(l:Integer[_], u:Expression<Integer[_]>) -> IndependentUniformInteger {
  return Uniform(Boxed(l), u);
}

/**
 * Create multivariate uniform distribution over integers.
 */
function Uniform(l:Integer[_], u:Integer[_]) -> IndependentUniformInteger {
  return Uniform(Boxed(l), Boxed(u));
}
