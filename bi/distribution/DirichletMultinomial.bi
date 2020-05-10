/**
 * Dirichlet-multinomial distribution.
 */
final class DirichletMultinomial(n:Expression<Integer>, ρ:Dirichlet) <
    Distribution<Integer[_]> {
  /**
   * Number of trials.
   */
  n:Expression<Integer> <- n;
   
  /**
   * Category probabilities.
   */
  ρ:Dirichlet <- ρ;

  function simulate() -> Integer[_] {
    return simulate_dirichlet_multinomial(n.value(), ρ.α.value());
  }
  
  function logpdf(x:Integer[_]) -> Real {
    return logpdf_dirichlet_multinomial(x, n.value(), ρ.α.value());
  }

  function update(x:Integer[_]) {
    ρ.α <- update_dirichlet_multinomial(x, n.value(), ρ.α.value());
  }

  function downdate(x:Integer[_]) {
    ρ.α <- downdate_dirichlet_multinomial(x, n.value(), ρ.α.value());
  }

  function link() {
    ρ.setChild(this);
  }
  
  function unlink() {
    ρ.releaseChild();
  }
}

function DirichletMultinomial(n:Expression<Integer>, ρ:Dirichlet) ->
    DirichletMultinomial {
  m:DirichletMultinomial(n, ρ);
  m.link();
  return m;
}
