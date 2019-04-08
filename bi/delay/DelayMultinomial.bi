/*
 * Delayed multinomial random variate.
 */
class DelayMultinomial(x:Random<Integer[_]>&, n:Integer, ρ:Real[_]) <
    DelayValue<Integer[_]>(x) {
  /**
   * Number of trials.
   */
  n:Integer <- n;
   
  /**
   * Category probabilities.
   */
  ρ:Real[_] <- ρ;

  function simulate() -> Integer[_] {
    return simulate_multinomial(n, ρ);
  }
  
  function observe(x:Integer[_]) -> Real {
    return observe_multinomial(x, n, ρ);
  }

  function update(x:Integer[_]) {
    //
  }

  function downdate(x:Integer[_]) {
    //
  }

  function pmf(x:Integer[_]) -> Real {
    return pmf_multinomial(x, n, ρ);
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "Multinomial");
    buffer.set("n", n);
    buffer.set("ρ", ρ);
  }
}

function DelayMultinomial(x:Random<Integer[_]>&, n:Integer, ρ:Real[_]) ->
    DelayMultinomial {
  m:DelayMultinomial(x, n, ρ);
  return m;
}
