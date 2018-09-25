/**
 * Multivariate uniform distribution.
 */
class MultivariateUniform(l:Expression<Real[_]>, u:Expression<Real[_]>) < Distribution<Real[_]> {
  /**
   * Lower bound.
   */
  l:Expression<Real[_]> <- l;
  
  /**
   * Upper bound.
   */
  u:Expression<Real[_]> <- u;

  function graft() {
    if delay? {
      delay!.prune();
    } else {
      delay <- DelayMultivariateUniform(x, l, u);
    }
  }
}

/**
 * Create multivariate uniform distribution.
 */
function Uniform(l:Expression<Real[_]>, u:Expression<Real[_]>) -> MultivariateUniform {
  m:MultivariateUniform(l, u);
  return m;
}

/**
 * Create multivariate uniform distribution.
 */
function Uniform(l:Expression<Real[_]>, u:Real[_]) -> MultivariateUniform {
  return Uniform(l, Boxed(u));
}

/**
 * Create multivariate uniform distribution.
 */
function Uniform(l:Real[_], u:Expression<Real[_]>) -> MultivariateUniform {
  return Uniform(Boxed(l), u);
}

/**
 * Create multivariate uniform distribution.
 */
function Uniform(l:Real[_], u:Real[_]) -> MultivariateUniform {
  return Uniform(Boxed(l), Boxed(u));
}
