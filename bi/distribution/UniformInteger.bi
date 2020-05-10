/**
 * Uniform integer distribution.
 */
final class UniformInteger(l:Expression<Integer>, u:Expression<Integer>) <
    BoundedDiscrete {
  /**
   * Lower bound.
   */
  l:Expression<Integer> <- l;

  /**
   * Upper bound.
   */
  u:Expression<Integer> <- u;
    
  function simulate() -> Integer {
    if value? {
      return value!;
    } else {
      return simulate_uniform_int(l.value(), u.value());
    }
  }

  function logpdf(x:Integer) -> Real {
    return logpdf_uniform_int(x, l.value(), u.value());
  }

  function cdf(x:Integer) -> Real? {
    return cdf_uniform_int(x, l.value(), u.value());
  }

  function quantile(P:Real) -> Integer? {
    return quantile_uniform_int(P, l.value(), u.value());
  }
  
  function lower() -> Integer? {
    return l.value();
  }
  
  function upper() -> Integer? {
    return u.value();
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "UniformInteger");
    buffer.set("l", l);
    buffer.set("u", u);
  }
}

/**
 * Create uniform distribution over integers.
 */
function Uniform(l:Expression<Integer>, u:Expression<Integer>) ->
    UniformInteger {
  m:UniformInteger(l, u);
  return m;
}

/**
 * Create uniform distribution over integers.
 */
function Uniform(l:Expression<Integer>, u:Integer) -> UniformInteger {
  return Uniform(l, Boxed(u));
}

/**
 * Create uniform distribution over integers.
 */
function Uniform(l:Integer, u:Expression<Integer>) -> UniformInteger {
  return Uniform(Boxed(l), u);
}

/**
 * Create uniform distribution over integers.
 */
function Uniform(l:Integer, u:Integer) -> UniformInteger {
  return Uniform(Boxed(l), Boxed(u));
}
