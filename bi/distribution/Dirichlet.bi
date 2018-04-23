/**
 * Dirichlet distribution.
 */
class Dirichlet(α:Expression<Real[_]>) < Random<Real[_]> {
  /**
   * Concentration.
   */
  α:Expression<Real[_]>;

  /**
   * Updated concentrations.
   */
  α_p:Real[_];

  function isDirichlet() -> Boolean {
    return isMissing();
  }

  function getDirichlet() -> Real[_] {
    return α_p;
  }

  function setDirichlet(θ:Real[_]) {
    α_p <- θ;
  }

  function doMarginalize() {
    α_p <- α.value();
  }

  function doSimulate() -> Real[_] {
    return simulate_dirichlet(α_p);
  }
  
  function doObserve(x:Real[_]) -> Real {
    return observe_dirichlet(x, α_p);
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
