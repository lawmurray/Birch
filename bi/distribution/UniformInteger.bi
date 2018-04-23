/**
 * Uniform distribution over integers.
 */
class UniformInteger(l:Expression<Integer>, u:Expression<Integer>) < Random<Integer> {
  /**
   * Lower bound.
   */
  l:Expression<Integer> <- l;
  
  /**
   * Upper bound.
   */
  u:Expression<Integer> <- u;

  function doSimulate() -> Integer {
    return simulate_int_uniform(l.value(), u.value());
  }
  
  function doObserve(x:Integer) -> Real {
    return observe_int_uniform(x, l.value(), u.value());
  }
}

/**
 * Create uniform distribution over integers.
 */
function Uniform(l:Expression<Integer>, u:Expression<Integer>) -> UniformInteger {
  m:UniformInteger(l, u);
  m.initialize();
  return m;
}

/**
 * Create uniform distribution over integers.
 */
function Uniform(l:Expression<Integer>, u:Integer) -> UniformInteger {
  return Uniform(l, Literal(u));
}

/**
 * Create uniform distribution over integers.
 */
function Uniform(l:Integer, u:Expression<Integer>) -> UniformInteger {
  return Uniform(Literal(l), u);
}

/**
 * Create uniform distribution over integers.
 */
function Uniform(l:Integer, u:Integer) -> UniformInteger {
  return Uniform(Literal(l), Literal(u));
}
