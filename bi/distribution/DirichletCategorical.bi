/*
 * ed Dirichlet-categorical random variate.
 */
final class DirichletCategorical(future:Integer?, futureUpdate:Boolean,
    ρ:Dirichlet) < Distribution<Integer>(future, futureUpdate) {
  /**
   * Category probabilities.
   */
  ρ:Dirichlet& <- ρ;

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

function DirichletCategorical(future:Integer?, futureUpdate:Boolean,
    ρ:Dirichlet) -> DirichletCategorical {
  m:DirichletCategorical(future, futureUpdate, ρ);
  ρ.setChild(m);
  return m;
}
