/**
 * Delta distribution, representing a distribution on a discrete space with
 * all probability mass at one location.
 */
class Delta(μ:Expression<Integer>) < Random<Integer> {
  /**
   * Location.
   */
  μ:Expression<Integer> <- μ;

  function doSimulate() -> Integer {
    return simulate_delta(μ.value());
  }
  
  function doObserve(x:Integer) -> Real {
    return observe_delta(x, μ.value());
  }
}

/**
 * Create delta distribution.
 *
 * - μ: Location.
 */
function Delta(μ:Expression<Integer>) -> Delta {
  m:Delta(μ);
  m.initialize();
  return m;
}

/**
 * Create delta distribution.
 *
 * - μ: Location.
 */
function Delta(μ:Integer) -> Delta {
  return Delta(Literal(μ));
}
