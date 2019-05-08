/*
 * Delayed Dirichlet-categorical random variate.
 */
final class DelayDirichletCategorical(future:Integer?, futureUpdate:Boolean,
    ρ:DelayDirichlet) < DelayValue<Integer>(future, futureUpdate) {
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

  function write(buffer:Buffer) {
    buffer.set(value());
  }
}

function DelayDirichletCategorical(future:Integer?, futureUpdate:Boolean,
    ρ:DelayDirichlet) -> DelayDirichletCategorical {
  m:DelayDirichletCategorical(future, futureUpdate, ρ);
  ρ.setChild(m);
  return m;
}
