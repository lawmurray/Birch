/*
 * Delayed multinomial random variate.
 */
class DelayMultinomial(n:Integer, ρ:Real[_]) < DelayValue<Integer[_]> {
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

  function pmf(x:Integer[_]) -> Real {
    return pmf_multinomial(x, n, ρ);
  }
}

function DelayMultinomial(n:Integer, ρ:Real[_]) -> DelayMultinomial {
  m:DelayMultinomial(n, ρ);
  return m;
}
