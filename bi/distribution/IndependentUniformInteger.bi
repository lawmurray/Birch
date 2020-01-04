/*
 * ed multivariate uniform random variable over integers.
 */
final class IndependentUniformInteger(future:Integer[_]?,
    futureUpdate:Boolean, l:Expression<Integer[_]>,
    u:Expression<Integer[_]>) <
    Distribution<Integer[_]>(future, futureUpdate) {
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
    return simulate_independent_uniform_int(l, u);
  }
  
  function logpdf(x:Integer[_]) -> Real {
    return logpdf_independent_uniform_int(x, l, u);
  }

  function graft() {
    if delay? {
      delay!.prune();
    } else {
      delay <- IndependentUniformInteger(future, futureUpdate, l, u);
    }
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "IndependentUniformInteger");
    buffer.set("l", l);
    buffer.set("u", u);
  }
}

function IndependentUniformInteger(future:Integer[_]?,
    futureUpdate:Boolean, l:Integer[_], u:Integer[_]) ->
    IndependentUniformInteger {
  m:IndependentUniformInteger(future, futureUpdate, l, u);
  return m;
}

/**
 * Create multivariate uniform distribution over integers.
 */
function Uniform(l:Expression<Integer[_]>, u:Expression<Integer[_]>) -> IndependentUniformInteger {
  assert l.rows() == u.rows();
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
