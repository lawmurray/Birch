/*
 * Delayed Dirichlet-categorical random variate.
 */
class DelayDirichletCategorical(x:Random<Integer>&, ρ:DelayDirichlet) <
    DelayValue<Integer>(x) {
  /**
   * Category probabilities.
   */
  ρ:DelayDirichlet& <- ρ;

  function simulate() -> Integer {
    return simulate_dirichlet_categorical(ρ!.α);
  }
  
  function observe(x:Integer) -> Real {
    return observe_dirichlet_categorical(x, ρ!.α);
  }

  function update(x:Integer) {
    ρ!.α <- update_dirichlet_categorical(x, ρ!.α);
  }

  function downdate(x:Integer) {
    ρ!.α <- downdate_dirichlet_categorical(x, ρ!.α);
  }

  function pmf(x:Integer) -> Real {
    return pmf_dirichlet_categorical(x, ρ!.α);
  }

  function cdf(x:Integer) -> Real {
    return cdf_dirichlet_categorical(x, ρ!.α);
  }
}

function DelayDirichletCategorical(x:Random<Integer>&, ρ:DelayDirichlet) ->
    DelayDirichletCategorical {
  m:DelayDirichletCategorical(x, ρ);
  ρ.setChild(m);
  return m;
}
