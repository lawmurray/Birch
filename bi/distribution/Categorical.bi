/**
 * Categorical distribution.
 */
class Categorical(ρ:Expression<Real[_]>) < Random<Integer> {
  /**
   * Category probabilities.
   */
  ρ:Expression<Real[_]> <- ρ;

  function doSimulate()-> Integer {
    return simulate_categorical(ρ.value());
  }
  
  function doObserve(x:Integer) -> Real {
    return observe_categorical(x, ρ.value());
  }
}

/**
 * Create categorical distribution.
 */
function Categorical(ρ:Expression<Real[_]>) -> Categorical {
  m:Categorical(ρ);
  m.initialize();
  return m;
}

/**
 * Create categorical distribution.
 */
function Categorical(ρ:Real[_]) -> Categorical {
  return Categorical(Literal(ρ));
}
