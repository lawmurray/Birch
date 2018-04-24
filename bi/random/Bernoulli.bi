/**
 * Bernoulli distribution.
 */
class Bernoulli(ρ:Expression<Real>) < Random<Boolean> {
  /**
   * Probability of a true result.
   */
  ρ:Expression<Real> <- ρ;

  function doSimulate() -> Boolean {
    return simulate_bernoulli(ρ.value());
  }
  
  function doObserve(x:Boolean) -> Real {
    return observe_bernoulli(x, ρ.value());
  }
}

/**
 * Create Bernoulli distribution.
 */
function Bernoulli(ρ:Expression<Real>) -> Bernoulli {
  m:Bernoulli(ρ);
  return m;
}

/**
 * Create Bernoulli distribution.
 */
function Bernoulli(ρ:Real) -> Bernoulli {
  return Bernoulli(Literal(ρ));
}
