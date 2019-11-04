/*
 * Delayed Dirichlet random variate.
 */
final class DelayDirichlet(future:Real[_]?, futureUpdate:Boolean, α:Real[_])
    < DelayValue<Real[_]>(future, futureUpdate) {
  /**
   * Concentrations.
   */
  α:Real[_] <- α;

  function simulate() -> Real[_] {
    return simulate_dirichlet(α);
  }
  
  function logpdf(x:Real[_]) -> Real {
    return logpdf_dirichlet(x, α);
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "Dirichlet");
    buffer.set("α", α);
  }
}

function DelayDirichlet(future:Real[_]?, futureUpdate:Boolean, α:Real[_]) ->
    DelayDirichlet {
  m:DelayDirichlet(future, futureUpdate, α);
  return m;
}
