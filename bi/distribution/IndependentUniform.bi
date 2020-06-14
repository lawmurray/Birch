/**
 * Multivariate uniform distribution where each element is independent.
 */
final class IndependentUniform(l:Expression<Real[_]>,
    u:Expression<Real[_]>) < Distribution<Real[_]> {
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

  function supportsLazy() -> Boolean {
    return false;
  }

  function simulate() -> Real[_] {
    return simulate_independent_uniform(l.value(), u.value());
  }
  
//  function simulateLazy() -> Real[_]? {
//    return simulate_independent_uniform(l.get(), u.get());
//  }

  function logpdf(x:Real[_]) -> Real {
    return logpdf_independent_uniform(x, l.value(), u.value());
  }

//  function logpdfLazy(x:Expression<Real[_]>) -> Expression<Real>? {
//    return logpdf_lazy_independent_uniform(x, l, u);
//  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "IndependentUniform");
    buffer.set("l", l);
    buffer.set("u", u);
  }
}

/**
 * Create multivariate uniform distribution where each element is
 * independent.
 */
function Uniform(l:Expression<Real[_]>, u:Expression<Real[_]>) ->
    IndependentUniform {
  m:IndependentUniform(l, u);
  return m;
}

/**
 * Create multivariate uniform distribution where each element is
 * independent.
 */
function Uniform(l:Expression<Real[_]>, u:Real[_]) -> IndependentUniform {
  return Uniform(l, box(u));
}

/**
 * Create multivariate uniform distribution where each element is
 * independent.
 */
function Uniform(l:Real[_], u:Expression<Real[_]>) -> IndependentUniform {
  return Uniform(box(l), u);
}

/**
 * Create multivariate uniform distribution where each element is
 * independent.
 */
function Uniform(l:Real[_], u:Real[_]) -> IndependentUniform {
  return Uniform(box(l), box(u));
}
