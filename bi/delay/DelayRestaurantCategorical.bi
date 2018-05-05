/**
 * Delayed restaurant-categorical random variate.
 */
class DelayRestaurantCategorical(x:Random<Integer>, ρ:DelayRestaurant) <
    DelayValue<Integer>(x) {
  /**
   * Category probabilities.
   */
  ρ:DelayRestaurant <- ρ;

  function doSimulate() -> Integer {
    return simulate_crp_categorical(ρ.α, ρ.θ, ρ.n, ρ.N);
  }
  
  function doObserve(x:Integer) -> Real {
    return observe_crp_categorical(x, ρ.α, ρ.θ, ρ.n, ρ.N);
  }

  function doCondition(x:Integer) {
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
}

function DelayRestaurantCategorical(x:Random<Integer>, ρ:DelayRestaurant) ->
    DelayRestaurantCategorical {
  m:DelayRestaurantCategorical(x, ρ);
  return m;
}
