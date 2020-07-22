/**
 * Dirichlet-categorical distribution.
 */
final class DirichletCategorical(ρ:Dirichlet) < Distribution<Integer> {
  /**
   * Category probabilities.
   */
  ρ:Dirichlet <- ρ;

  function supportsLazy() -> Boolean {
    return false;
  }

  function simulate() -> Integer {
    return simulate_dirichlet_categorical(ρ.α.value());
  }

//  function simulateLazy() -> Integer? {
//    return simulate_dirichlet_categorical(ρ.α.get());
//  }
  
  function logpdf(x:Integer) -> Real {
    return logpdf_dirichlet_categorical(x, ρ.α.value());
  }

//  function logpdfLazy(x:Expression<Integer>) -> Expression<Real>? {
//    return logpdf_lazy_dirichlet_categorical(x, ρ.α);
//  }

  function update(x:Integer) {
    ρ.α <- box(update_dirichlet_categorical(x, ρ.α.value()));
  }

//  function updateLazy(x:Expression<Integer>) {
//    ρ.α <- update_lazy_dirichlet_categorical(x, ρ.α);
//  }

  function downdate(x:Integer) {
    ρ.α <- box(downdate_dirichlet_categorical(x, ρ.α.value()));
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
    ρ.releaseChild(this);
  }
}

function DirichletCategorical(ρ:Dirichlet) -> DirichletCategorical {
  m:DirichletCategorical(ρ);
  m.link();
  return m;
}
