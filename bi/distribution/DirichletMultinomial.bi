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
  ρ:Dirichlet& <- ρ;

  function supportsLazy() -> Boolean {
    return false;
  }

  function simulate() -> Integer[_] {
    auto ρ <- this.ρ;
    return simulate_dirichlet_multinomial(n.value(), ρ.α.value());
  }

//  function simulateLazy() -> Integer[_]? {
//    auto ρ <- this.ρ;
//    return simulate_dirichlet_multinomial(n.get(), ρ.α.get());
//  }
  
  function logpdf(x:Integer[_]) -> Real {
    auto ρ <- this.ρ;
    return logpdf_dirichlet_multinomial(x, n.value(), ρ.α.value());
  }

//  function logpdfLazy(x:Expression<Integer[_]>) -> Expression<Real>? {
//    auto ρ <- this.ρ;
//    return logpdf_lazy_dirichlet_multinomial(x, n, ρ.α);
//  }

  function update(x:Integer[_]) {
    auto ρ <- this.ρ;
    ρ.α <- box(update_dirichlet_multinomial(x, n.value(), ρ.α.value()));
  }

//  function updateLazy(x:Expression<Integer[_]>) {
//    auto ρ <- this.ρ;
//    ρ.α <- update_lazy_dirichlet_multinomial(x, n, ρ.α);
//  }

  function downdate(x:Integer[_]) {
    auto ρ <- this.ρ;
    ρ.α <- box(downdate_dirichlet_multinomial(x, n.value(), ρ.α.value()));
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

function DirichletMultinomial(n:Expression<Integer>, ρ:Dirichlet) ->
    DirichletMultinomial {
  m:DirichletMultinomial(n, ρ);
  m.link();
  return m;
}
