/*
 * ed uniform random variable.
 */
final class Uniform(future:Real?, futureUpdate:Boolean, l:Expression<Real>, u:Expression<Real>)
    < Distribution<Real>(future, futureUpdate) {
  /**
   * Lower bound.
   */
  l:Expression<Real> <- l;
  
  /**
   * Upper bound.
   */
  u:Expression<Real> <- u;

  function simulate() -> Real {
    return simulate_uniform(l, u);
  }
  
  function logpdf(x:Real) -> Real {
    return logpdf_uniform(x, l, u);
  }

  function cdf(x:Real) -> Real? {
    return cdf_uniform(x, l, u);
  }

  function quantile(p:Real) -> Real? {
    return quantile_uniform(p, l, u);
  }

  function lower() -> Real? {
    return l.value();
  }
  
  function upper() -> Real? {
    return u.value();
  }

  function graft() -> Distribution<Real> {
    prune();
    return this;
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "Uniform");
    buffer.set("l", l);
    buffer.set("u", u);
  }
}

function Uniform(future:Real?, futureUpdate:Boolean, l:Expression<Real>,
    u:Expression<Real>) -> Uniform {
  m:Uniform(future, futureUpdate, l, u);
  return m;
}

/**
 * Create a uniform distribution.
 */
function Uniform(l:Expression<Real>, u:Expression<Real>) -> Uniform {
  m:Uniform(nil, true, l, u);
  return m;
}

/**
 * Create a uniform distribution.
 */
function Uniform(l:Expression<Real>, u:Real) -> Uniform {
  return Uniform(l, Boxed(u));
}

/**
 * Create a uniform distribution.
 */
function Uniform(l:Real, u:Expression<Real>) -> Uniform {
  return Uniform(Boxed(l), u);
}

/**
 * Create a uniform distribution.
 */
function Uniform(l:Real, u:Real) -> Uniform {
  return Uniform(Boxed(l), Boxed(u));
}
