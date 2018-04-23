/**
 * Dirichlet distribution.
 */
class Dirichlet(α:Expression<Real[_]>) < Random<Real[_]> {
  /**
   * Concentration.
   */
  α:Expression<Real[_]>;

  function doSimulate() -> Real[_] {
    return simulate_dirichlet(α.value());
  }
  
  function doObserve(x:Real[_]) -> Real {
    return observe_dirichlet(x, α.value());
  }
}

/**
 * Create Dirichlet distribution.
 */
function Dirichlet(α:Expression<Real[_]>) -> Dirichlet {
  m:Dirichlet(α);
  m.initialize();
  return m;
}

/**
 * Create Dirichlet distribution.
 */
function Dirichlet(α:Real[_]) -> Dirichlet {
  return Dirichlet(Literal(α));
}
