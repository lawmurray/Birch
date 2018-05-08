/*
 * Delayed Dirichlet random variate.
 */
class DelayDirichlet(α:Real[_]) < DelayValue<Real[_]> {
  /**
   * Concentrations.
   */
  α:Real[_] <- α;

  function simulate() -> Real[_] {
    return simulate_dirichlet(α);
  }
  
  function observe(x:Real[_]) -> Real {
    return observe_dirichlet(x, α);
  }

  function pdf(x:Real[_]) -> Real {
    return pdf_dirichlet(x, α);
  }
}

function DelayDirichlet(α:Real[_]) -> DelayDirichlet {
  m:DelayDirichlet(α);
  return m;
}
