/**
 * Dirichlet distribution.
 */
class Dirichlet(α:Expression<Real[_]>) < Random<Real[_]> {
  /**
   * Concentration.
   */
  α:Expression<Real[_]>;

  /**
   * Delayed sampling node.
   */
  delay:DelayDirichlet?;

  function isDirichlet() -> Boolean {
    return isMissing();
  }

  function getDirichlet() -> DelayDirichlet {
    if (!delay?) {
      delay:DelayDirichlet(this, α.value());
      this.delay <- delay;
    }
    return delay!;
  }

  function doSimulate() -> Real[_] {
    return delay!.doSimulate();
  }
  
  function doObserve(x:Real[_]) -> Real {
    return delay!.doObserve(x);
  }
}

/**
 * Create Dirichlet distribution.
 */
function Dirichlet(α:Expression<Real[_]>) -> Dirichlet {
  m:Dirichlet(α);
  return m;
}

/**
 * Create Dirichlet distribution.
 */
function Dirichlet(α:Real[_]) -> Dirichlet {
  return Dirichlet(Literal(α));
}
