/*
 * Delayed Bernoulli random variate.
 */
final class DelayBernoulli(future:Boolean?, futureUpdate:Boolean, ρ:Real) <
    DelayValue<Boolean>(future, futureUpdate) {
  /**
   * Success probability.
   */
  ρ:Real <- ρ;

  function simulate() -> Boolean {
    return simulate_bernoulli(ρ);
  }
  
  function observe(x:Boolean) -> Real {
    return observe_bernoulli(x, ρ);
  }

  function update(x:Boolean) {
    //
  }

  function downdate(x:Boolean) {
    //
  }

  function pmf(x:Boolean) -> Real {
    return pmf_bernoulli(x, ρ);
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "Bernoulli");
    buffer.set("ρ", ρ);
  }
}

function DelayBernoulli(future:Boolean?, futureUpdate:Boolean, ρ:Real) ->
    DelayBernoulli {
  m:DelayBernoulli(future, futureUpdate, ρ);
  return m;
}
