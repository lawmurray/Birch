/**
 * Dirichlet-categorical distribution.
 */
final class DirichletCategorical(ρ:Dirichlet) < Distribution<Integer> {
  /**
   * Category probabilities.
   */
  ρ:Dirichlet& <- ρ;

  function supportsLazy() -> Boolean {
    return false;
  }

  function simulate() -> Integer {
    auto ρ <- this.ρ;
    return simulate_dirichlet_categorical(ρ.α.value());
  }

//  function simulateLazy() -> Integer? {
//    auto ρ <- this.ρ;
//    return simulate_dirichlet_categorical(ρ.α.get());
//  }
  
  function logpdf(x:Integer) -> Real {
    auto ρ <- this.ρ;
    return logpdf_dirichlet_categorical(x, ρ.α.value());
  }

//  function logpdfLazy(x:Expression<Integer>) -> Expression<Real>? {
//    auto ρ <- this.ρ;
//    return logpdf_lazy_dirichlet_categorical(x, ρ.α);
//  }

  function update(x:Integer) {
    auto ρ <- this.ρ;
    ρ.α <- box(update_dirichlet_categorical(x, ρ.α.value()));
  }

//  function updateLazy(x:Expression<Integer>) {
//    auto ρ <- this.ρ;
//    ρ.α <- update_lazy_dirichlet_categorical(x, ρ.α);
//  }

  function downdate(x:Integer) {
    auto ρ <- this.ρ;
    ρ.α <- box(downdate_dirichlet_categorical(x, ρ.α.value()));
  }

  function lower() -> Integer? {
    return 1;
  }

  function upper() -> Integer? {
    auto ρ <- this.ρ;
    return ρ.α.rows();
  }

  function link() {
    auto ρ <- this.ρ;
    ρ.setChild(this);
  }
  
  function unlink() {
    auto ρ <- this.ρ;
    ρ.releaseChild(this);
  }
}

function DirichletCategorical(ρ:Dirichlet) -> DirichletCategorical {
  m:DirichletCategorical(ρ);
  m.link();
  return m;
}
