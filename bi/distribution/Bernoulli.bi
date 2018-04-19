/**
 * Bernoulli distribution.
 */
class Bernoulli<Type1>(ρ:Type1) < Random<Boolean> {
  /**
   * Probability of a true result.
   */
  ρ:Type1 <- ρ;

  function update(ρ:Type1) {
    this.ρ <- ρ;
  }

  function doSimulate() -> Boolean {
    return simulate_bernoulli(ρ);
  }
  
  function doObserve(x:Boolean) -> Real {
    return observe_bernoulli(x, ρ);
  }
}

/**
 * Create Bernoulli distribution.
 */
function Bernoulli(ρ:Real) -> Bernoulli<Real> {
  m:Bernoulli<Real>(ρ);
  m.initialize();
  return m;
}

/**
 * Create Bernoulli distribution.
 */
function Bernoulli(ρ:Expression<Real>) -> Bernoulli<Expression<Real>> {
  m:Bernoulli<Expression<Real>>(ρ);
  m.initialize();
  return m;
}
