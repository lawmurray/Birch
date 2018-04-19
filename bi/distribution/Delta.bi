/**
 * Delta distribution, representing a distribution on a discrete space with
 * all probability mass at one location.
 */
class Delta<Type1>(μ:Type1) < Random<Integer> {
  /**
   * Location.
   */
  μ:Integer <- μ;

  function update(μ:Type1) {
    this.μ <- μ;
  }

  function doSimulate() -> Integer {
    return simulate_delta(global.value(μ));
  }
  
  function doObserve(x:Integer) -> Real {
    return observe_delta(x, global.value(μ));
  }
}

/**
 * Create delta distribution.
 *
 * - μ: Location.
 */
function Delta(μ:Integer) -> Delta<Integer> {
  m:Delta<Integer>(μ);
  m.initialize();
  return m;
}

/**
 * Create delta distribution.
 *
 * - μ: Location.
 */
function Delta(μ:Expression<Integer>) -> Delta<Expression<Integer>> {
  m:Delta<Expression<Integer>>(μ);
  m.initialize();
  return m;
}
