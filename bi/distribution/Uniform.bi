/**
 * Uniform distribution.
 */
final class Uniform(l:Expression<Real>, u:Expression<Real>) <
    Distribution<Real> {
  /**
   * Lower bound.
   */
  l:Expression<Real> <- l;
  
  /**
   * Upper bound.
   */
  u:Expression<Real> <- u;

  function simulate() -> Real {
    return simulate_uniform(l.value(), u.value());
  }
  
  function logpdf(x:Real) -> Real {
    return logpdf_uniform(x, l.value(), u.value());
  }

  function cdf(x:Real) -> Real? {
    return cdf_uniform(x, l.value(), u.value());
  }

  function quantile(P:Real) -> Real? {
    return quantile_uniform(P, l.value(), u.value());
  }

  function lower() -> Real? {
    return l.value();
  }
  
  function upper() -> Real? {
    return u.value();
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "Uniform");
    buffer.set("l", l);
    buffer.set("u", u);
  }
}

/**
 * Create a uniform distribution.
 */
function Uniform(l:Expression<Real>, u:Expression<Real>) -> Uniform {
  m:Uniform(l, u);
  return m;
}

/**
 * Create a uniform distribution.
 */
function Uniform(l:Expression<Real>, u:Real) -> Uniform {
  return Uniform(l, box(u));
}

/**
 * Create a uniform distribution.
 */
function Uniform(l:Real, u:Expression<Real>) -> Uniform {
  return Uniform(box(l), u);
}

/**
 * Create a uniform distribution.
 */
function Uniform(l:Real, u:Real) -> Uniform {
  return Uniform(box(l), box(u));
}
