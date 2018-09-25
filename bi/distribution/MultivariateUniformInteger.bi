/**
 * Multivariate uniform distribution over integers.
 */
class MultivariateUniformInteger(l:Expression<Integer[_]>, u:Expression<Integer[_]>) < Distribution<Integer[_]> {
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
      delay <- DelayMultivariateUniformInteger(x, l, u);
    }
  }
}

/**
 * Create multivariate uniform distribution over integers.
 */
function Uniform(l:Expression<Integer[_]>, u:Expression<Integer[_]>) -> MultivariateUniformInteger {
  m:MultivariateUniformInteger(l, u);
  return m;
}

/**
 * Create multivariate uniform distribution over integers.
 */
function Uniform(l:Expression<Integer[_]>, u:Integer[_]) -> MultivariateUniformInteger {
  return Uniform(l, Boxed(u));
}

/**
 * Create multivariate uniform distribution over integers.
 */
function Uniform(l:Integer[_], u:Expression<Integer[_]>) -> MultivariateUniformInteger {
  return Uniform(Boxed(l), u);
}

/**
 * Create multivariate uniform distribution over integers.
 */
function Uniform(l:Integer[_], u:Integer[_]) -> MultivariateUniformInteger {
  return Uniform(Boxed(l), Boxed(u));
}
