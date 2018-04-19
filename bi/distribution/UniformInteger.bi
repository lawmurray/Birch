/**
 * Uniform distribution over integers.
 */
class UniformInteger<Type1,Type2>(l:Type1, u:Type2) < Random<Integer> {
  /**
   * Lower bound.
   */
  l:Type1 <- l;
  
  /**
   * Upper bound.
   */
  u:Type2 <- u;

  function update(l:Type1, u:Type2) {
    this.l <- l;
    this.u <- u;
  }

  function doRealize() -> Integer {
    return simulate_int_uniform(global.value(l), global.value(u));
  }
  
  function doObserve(x:Integer) -> Real {
    return observe_int_uniform(x, global.value(l), global.value(u));
  }
}

/**
 * Create uniform distribution over integers.
 */
function Uniform(l:Integer, u:Integer) -> UniformInteger<Integer,Integer> {
  m:UniformInteger<Integer,Integer>(l, u);
  m.initialize();
  return m;
}

/**
 * Create uniform distribution over integers.
 */
function Uniform(l:Expression<Integer>, u:Integer) ->
    UniformInteger<Expression<Integer>,Integer> {
  m:UniformInteger<Expression<Integer>,Integer>(l, u);
  m.initialize();
  return m;
}

/**
 * Create uniform distribution over integers.
 */
function Uniform(l:Integer, u:Expression<Integer>) ->
    UniformInteger<Integer,Expression<Integer>> {
  m:UniformInteger<Integer,Expression<Integer>>(l, u);
  m.initialize();
  return m;
}

/**
 * Create uniform distribution over integers.
 */
function Uniform(l:Expression<Integer>, u:Expression<Integer>) ->
    UniformInteger<Expression<Integer>,Expression<Integer>> {
  m:UniformInteger<Expression<Integer>,Expression<Integer>>(l, u);
  m.initialize();
  return m;
}
