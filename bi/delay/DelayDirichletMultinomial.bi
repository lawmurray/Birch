/*
 * Delayed Dirichlet-multinomial random variate.
 */
final class DelayDirichletMultinomial(future:Integer[_]?,
    futureUpdate:Boolean, n:Integer, ρ:DelayDirichlet) <
    DelayValue<Integer[_]>(future, futureUpdate) {
  /**
   * Number of trials.
   */
  n:Integer <- n;
   
  /**
   * Category probabilities.
   */
  ρ:DelayDirichlet& <- ρ;

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

  function pdf(x:Integer[_]) -> Real {
    return pdf_dirichlet_multinomial(x, n, ρ.α);
  }

  function write(buffer:Buffer) {
    buffer.set(value());
  }
}

function DelayDirichletMultinomial(future:Integer[_]?, futureUpdate:Boolean,
    n:Integer, ρ:DelayDirichlet) -> DelayDirichletMultinomial {
  m:DelayDirichletMultinomial(future, futureUpdate, n, ρ);
  ρ.setChild(m);
  return m;
}
