/**
 * Multivariate uniform distribution.
 */
final class MultivariateUniform(l:Expression<Real[_]>, u:Expression<Real[_]>) < Distribution<Real[_]> {
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

  function write(buffer:Buffer) {
    if delay? {
      delay!.write(buffer);
    } else {
      buffer.set("class", "MultivariateUniform");
      buffer.set("l", l.value());
      buffer.set("u", u.value());
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
