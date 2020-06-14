/**
 * Multivariate uniform distribution over integers where each element is
 * independent.
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

  function rows() -> Integer {
    return l.rows();
  }

  function supportsLazy() -> Boolean {
    return false;
  }

  function simulate() -> Integer[_] {
    return simulate_independent_uniform_int(l.value(), u.value());
  }

//  function simulateLazy() -> Integer[_]? {
//    return simulate_independent_uniform_int(l.get(), u.get());
//  }
  
  function logpdf(x:Integer[_]) -> Real {
    return logpdf_independent_uniform_int(x, l.value(), u.value());
  }

//  function logpdfLazy(x:Expression<Integer[_]>) -> Expression<Real>? {
//    return logpdf_lazy_independent_uniform_int(x, l, u);
//  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "IndependentUniformInteger");
    buffer.set("l", l);
    buffer.set("u", u);
  }
}

/**
 * Create multivariate uniform distribution over integers where each element
 * is independent.
 */
function Uniform(l:Expression<Integer[_]>, u:Expression<Integer[_]>) ->
    IndependentUniformInteger {
  m:IndependentUniformInteger(l, u);
  return m;
}

/**
 * Create multivariate uniform distribution over integers where each element
 * is independent.
 */
function Uniform(l:Expression<Integer[_]>, u:Integer[_]) ->
    IndependentUniformInteger {
  return Uniform(l, box(u));
}

/**
 * Create multivariate uniform distribution over integers where each element
 * is independent.
 */
function Uniform(l:Integer[_], u:Expression<Integer[_]>) ->
    IndependentUniformInteger {
  return Uniform(box(l), u);
}

/**
 * Create multivariate uniform distribution over integers where each element
 * is independent.
 */
function Uniform(l:Integer[_], u:Integer[_]) -> IndependentUniformInteger {
  return Uniform(box(l), box(u));
}
