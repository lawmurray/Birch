/*
 * ed Dirichlet-multinomial random variate.
 */
final class DirichletMultinomial(future:Integer[_]?,
    futureUpdate:Boolean, n:Integer, ρ:Dirichlet) <
    Distribution<Integer[_]>(future, futureUpdate) {
  /**
   * Number of trials.
   */
  n:Integer <- n;
   
  /**
   * Category probabilities.
   */
  ρ:Dirichlet& <- ρ;

  function simulate() -> Integer[_] {
    return simulate_dirichlet_multinomial(n, ρ.α);
  }
  
  function logpdf(x:Integer[_]) -> Real {
    return logpdf_dirichlet_multinomial(x, n, ρ.α);
  }

  function update(x:Integer[_]) {
    ρ.α <- update_dirichlet_multinomial(x, n, ρ.α);
  }

  function downdate(x:Integer[_]) {
    ρ.α <- downdate_dirichlet_multinomial(x, n, ρ.α);
  }
}

function DirichletMultinomial(future:Integer[_]?, futureUpdate:Boolean,
    n:Integer, ρ:Dirichlet) -> DirichletMultinomial {
  m:DirichletMultinomial(future, futureUpdate, n, ρ);
  ρ.setChild(m);
  return m;
}
