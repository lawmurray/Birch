/**
 * Categorical distribution.
 */
class Categorical<Type1>(ρ:Type1) < Random<Integer> {
  /**
   * Category probabilities.
   */
  ρ:Type1 <- ρ;

  function update(ρ:Type1) {
    this.ρ <- ρ;
  }

  function doSimulate()-> Integer {
    return simulate_categorical(global.value(ρ));
  }
  
  function doObserve(x:Integer) -> Real {
    return observe_categorical(x, global.value(ρ));
  }
}

/**
 * Create categorical distribution.
 */
function Categorical(ρ:Real[_]) -> Categorical<Real[_]> {
  m:Categorical<Real[_]>(ρ);
  m.initialize();
  return m;
}

/**
 * Create categorical distribution.
 */
function Categorical(ρ:Expression<Real[_]>) -> Categorical<Expression<Real[_]>> {
  m:Categorical<Expression<Real[_]>>(ρ);
  m.initialize();
  return m;
}
