/*
 * Dirichlet-categorical distribution.
 */
final class DirichletCategorical(ρ:Dirichlet) < Distribution<Integer> {
  /**
   * Category probabilities.
   */
  ρ:Dirichlet <- ρ;

  function simulate() -> Integer {
    return simulate_dirichlet_categorical(ρ.α.value());
  }
  
  function logpdf(x:Integer) -> Real {
    return logpdf_dirichlet_categorical(x, ρ.α.value());
  }

  function update(x:Integer) {
    ρ.α <- update_dirichlet_categorical(x, ρ.α.value());
  }

  function downdate(x:Integer) {
    ρ.α <- downdate_dirichlet_categorical(x, ρ.α.value());
  }

  function lower() -> Integer? {
    return 1;
  }

  function upper() -> Integer? {
    return ρ.α.rows();
  }

  function link() {
    ρ.setChild(this);
  }
  
  function unlink() {
    ρ.releaseChild();
  }
}

function DirichletCategorical(ρ:Dirichlet) -> DirichletCategorical {
  m:DirichletCategorical(ρ);
  m.link();
  return m;
}
