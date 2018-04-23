/**
 * Poisson distribution.
 */
class Poisson(λ:Expression<Real>) < Random<Integer> {
  /**
   * Rate.
   */
  λ:Expression<Real> <- λ;

  function doSimulate() -> Integer {
    return simulate_poisson(λ.value());
  }
  
  function doObserve(x:Integer) -> Real {
    return observe_poisson(x, λ.value());
  }
}

/**
 * Create Poisson distribution.
 */
function Poisson(λ:Expression<Real>) -> Poisson {
  m:Poisson(λ);
  m.initialize();
  return m;
}

/**
 * Create Poisson distribution.
 */
function Poisson(λ:Real) -> Poisson {
  return Poisson(Literal(λ));
}
