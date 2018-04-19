/**
 * Poisson distribution.
 */
class Poisson<Type1>(λ:Type1) < Random<Integer> {
  /**
   * Rate.
   */
  λ:Type1 <- λ;

  function update(λ:Type1) {
    this.λ <- λ;
  }

  function doSimulate() -> Integer {
    return simulate_poisson(global.value(λ));
  }
  
  function doObserve(x:Integer) -> Real {
    return observe_poisson(x, global.value(λ));
  }
}

/**
 * Create Poisson distribution.
 */
function Poisson(λ:Real) -> Poisson<Real> {
  m:Poisson<Real>(λ);
  m.initialize();
  return m;
}

/**
 * Create Poisson distribution.
 */
function Poisson(λ:Expression<Real>) -> Poisson<Expression<Real>> {
  m:Poisson<Expression<Real>>(λ);
  m.initialize();
  return m;
}
