/**
 * Uniform distribution over integers on an orthogonal lattice.
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

  function graft() {
    if delay? {
      delay!.prune();
    } else {
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
