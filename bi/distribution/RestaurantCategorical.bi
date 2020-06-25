/**
 * Restaurant-categorical distribution.
 */
final class RestaurantCategorical(ρ:Restaurant) < Distribution<Integer> {
  /**
   * Category probabilities.
   */
  ρ:Restaurant& <- ρ;

  function supportsLazy() -> Boolean {
    return false;
  }

  function simulate() -> Integer {
    auto ρ <- this.ρ;
    return simulate_crp_categorical(ρ.α.value(), ρ.θ.value(), ρ.n, ρ.N);
  }

//  function simulateLazy() -> Integer? {
//    auto ρ <- this.ρ;
//    return simulate_crp_categorical(ρ.α.get(), ρ.θ.get(), ρ.n, ρ.N);
//  }
  
  function logpdf(x:Integer) -> Real {
    auto ρ <- this.ρ;
    return logpdf_crp_categorical(x, ρ.α.value(), ρ.θ.value(), ρ.n, ρ.N);
  }

//  function logpdfLazy(x:Expression<Integer>) -> Expression<Real>? {
//    auto ρ <- this.ρ;
//    return logpdf_lazy_crp_categorical(x, ρ.α, ρ.θ, ρ.n, ρ.N);
//  }

  function update(x:Integer) {
    auto ρ <- this.ρ;
    //@todo use Vector with its in-place enlarge support?
    assert x <= ρ.K + 1;
    if (x == ρ.K + 1) {
      n1:Integer[ρ.K + 1];
      n1[1..ρ.K] <- ρ.n;
      n1[x] <- 1;
      ρ.n <- n1;
      ρ.K <- ρ.K + 1;
    } else {
      ρ.n[x] <- ρ.n[x] + 1;
    }
    ρ.N <- ρ.N + 1;
  }

//  function updateLazy(x:Expression<Integer>) {
//
//  }
  
  function link() {
    auto ρ <- this.ρ;
    ρ.setChild(this);
  }
  
  function unlink() {
    auto ρ <- this.ρ;
    ρ.releaseChild(this);
  }
}

function RestaurantCategorical(ρ:Restaurant) -> RestaurantCategorical {
  m:RestaurantCategorical(ρ);
  m.link();
  return m;
}
