/*
 * Delayed Dirichlet random variate.
 */
class DelayDirichlet(x:Random<Real[_]>, α:Real[_]) < DelayValue<Real[_]>(x) {
  /**
   * Concentrations.
   */
  α:Real[_] <- α;

  function doSimulate() -> Real[_] {
    return simulate_dirichlet(α);
  }
  
  function doObserve(x:Real[_]) -> Real {
    return observe_dirichlet(x, α);
  }
}

function DelayDirichlet(x:Random<Real[_]>, α:Real[_]) -> DelayDirichlet {
  m:DelayDirichlet(x, α);
  return m;
}
