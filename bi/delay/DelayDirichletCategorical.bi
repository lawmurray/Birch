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
    return simulate_dirichlet_categorical(ρ.α);
  }
  
  function logpdf(x:Integer) -> Real {
    return logpdf_dirichlet_categorical(x, ρ.α);
  }

  function update(x:Integer) {
    ρ.α <- update_dirichlet_categorical(x, ρ.α);
  }

  function downdate(x:Integer) {
    ρ.α <- downdate_dirichlet_categorical(x, ρ.α);
  }

  function lower() -> Integer? {
    return 1;
  }

  function upper() -> Integer? {
    return length(ρ.α);
  }
}

function DelayDirichletCategorical(future:Integer?, futureUpdate:Boolean,
    ρ:DelayDirichlet) -> DelayDirichletCategorical {
  m:DelayDirichletCategorical(future, futureUpdate, ρ);
  ρ.setChild(m);
  return m;
}
