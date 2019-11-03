/*
 * Delayed Beta random variate.
 */
final class DelayBeta(future:Real?, futureUpdate:Boolean, α:Real, β:Real) <
    DelayValue<Real>(future, futureUpdate) {
  /**
   * First shape.
   */
  α:Real <- α;

  /**
   * Second shape.
   */
  β:Real <- β;

  function simulate() -> Real {
    return simulate_beta(α, β);
  }
  
  function logpdf(x:Real) -> Real {
    return logpdf_beta(x, α, β);
  }

  function update(x:Real) {
    //
  }

  function downdate(x:Real) {
    //
  }

  function cdf(x:Real) -> Real {
    return cdf_beta(x, α, β);
  }

  function quantile(p:Real) -> Real? {
    return quantile_beta(p, α, β);
  }

  function lower() -> Real? {
    return 0.0;
  }
  
  function upper() -> Real? {
    return 1.0;
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "Beta");
    buffer.set("α", α);
    buffer.set("β", β);
  }
}

function DelayBeta(future:Real?, futureUpdate:Boolean, α:Real, β:Real) -> DelayBeta {
  m:DelayBeta(future, futureUpdate, α, β);
  return m;
}
