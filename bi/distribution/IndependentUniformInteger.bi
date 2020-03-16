/*
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

  function simulate() -> Integer[_] {
    return simulate_independent_uniform_int(l.value(), u.value());
  }
  
  function logpdf(x:Integer[_]) -> Real {
    return logpdf_independent_uniform_int(x, l.value(), u.value());
  }

  function graftFinalize() -> Boolean {
    l.value();
    u.value();
    return true;
  }

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
  return Uniform(l, Boxed(u));
}

/**
 * Create multivariate uniform distribution over integers where each element
 * is independent.
 */
function Uniform(l:Integer[_], u:Expression<Integer[_]>) ->
    IndependentUniformInteger {
  return Uniform(Boxed(l), u);
}

/**
 * Create multivariate uniform distribution over integers where each element
 * is independent.
 */
function Uniform(l:Integer[_], u:Integer[_]) -> IndependentUniformInteger {
  return Uniform(Boxed(l), Boxed(u));
}
