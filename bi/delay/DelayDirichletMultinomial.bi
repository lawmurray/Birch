/**
 * Delayed Dirichlet-multinomial random variate.
 */
class DelayDirichletMultinomial(x:Random<Integer[_]>, n:Integer,
    ρ:DelayDirichlet) < DelayValue<Integer[_]>(x) {
  /**
   * Number of trials.
   */
  n:Integer <- n;
   
  /**
   * Category probabilities.
   */
  ρ:DelayDirichlet <- ρ;

  function doSimulate() -> Integer[_] {
    return simulate_dirichlet_multinomial(n, ρ.α);
  }
  
  function doObserve(x:Integer[_]) -> Real {
    return observe_dirichlet_multinomial(x, n, ρ.α);
  }

  function doCondition(x:Integer[_]) {
    ρ.α <- update_dirichlet_multinomial(x, n, ρ.α);
  }
}

function DelayDirichletMultinomial(x:Random<Integer[_]>, n:Integer,
    ρ:DelayDirichlet) -> DelayDirichletMultinomial {
  m:DelayDirichletMultinomial(x, n, ρ);
  return m;
}
