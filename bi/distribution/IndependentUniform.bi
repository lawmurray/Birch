/*
 * ed multivariate uniform random variable.
 */
final class IndependentUniform(future:Real[_]?, futureUpdate:Boolean,
    l:Expression<Real[_]>, u:Expression<Real[_]>) < Distribution<Real[_]>(future, futureUpdate) {
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

  function simulate() -> Real[_] {
    return simulate_independent_uniform(l, u);
  }
  
  function logpdf(x:Real[_]) -> Real {
    return logpdf_independent_uniform(x, l, u);
  }

  function graft() -> Distribution<Real[_]> {
    prune();
    return this;
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "IndependentUniform");
    buffer.set("l", l);
    buffer.set("u", u);
  }
}

function IndependentUniform(future:Real[_]?, futureUpdate:Boolean,
    l:Expression<Real[_]>, u:Expression<Real[_]>) -> IndependentUniform {
  m:IndependentUniform(future, futureUpdate, l, u);
  return m;
}

/**
 * Create multivariate uniform distribution.
 */
function Uniform(l:Expression<Real[_]>, u:Expression<Real[_]>) -> IndependentUniform {
  assert l.rows() == u.rows();
  m:IndependentUniform(nil, true, l, u);
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
