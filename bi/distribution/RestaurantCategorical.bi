/**
 * Restaurant-categorical distribution.
 */
final class RestaurantCategorical(ρ:Restaurant) < Distribution<Integer> {
  /**
   * Category probabilities.
   */
  ρ:Restaurant <- ρ;

  function simulate() -> Integer {
    return simulate_crp_categorical(ρ.α.value(), ρ.θ.value(), ρ.n, ρ.N);
  }
  
  function logpdf(x:Integer) -> Real {
    return logpdf_crp_categorical(x, ρ.α.value(), ρ.θ.value(), ρ.n, ρ.N);
  }

  function update(x:Integer) {
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

  function link() {
    ρ.setChild(this);
  }
  
  function unlink() {
    ρ.releaseChild(this);
  }
}

function RestaurantCategorical(ρ:Restaurant) -> RestaurantCategorical {
  m:RestaurantCategorical(ρ);
  m.link();
  return m;
}
