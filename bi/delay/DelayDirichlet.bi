/*
 * Delayed Dirichlet random variate.
 */
class DelayDirichlet(x:Random<Real[_]>&, α:Real[_]) <
    DelayValue<Real[_]>(x) {
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

  function update(x:Real[_]) {
    //
  }

  function downdate(x:Real[_]) {
    //
  }

  function pdf(x:Real[_]) -> Real {
    return pdf_dirichlet(x, α);
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "Dirichlet");
    buffer.set("α", α);
  }
}

function DelayDirichlet(x:Random<Real[_]>&, α:Real[_]) -> DelayDirichlet {
  m:DelayDirichlet(x, α);
  return m;
}
