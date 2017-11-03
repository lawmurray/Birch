/**
 * Categorical with conjugate restaurant process prior on category
 * probabilities.
 */
class RestaurantProcessCategorical < Random<Integer> {
  /**
   * Category probabilities.
   */
  ρ:RestaurantProcess;

  function initialize(ρ:RestaurantProcess) {
    super.initialize(ρ);
    this.ρ <- ρ;
  }
  
  function doMarginalize() {

  }
  
  function doCondition() {
    ρ.update(x);
  }

  function doRealize() {
    if (isMissing()) {
      set(simulate_crp_categorical(ρ.α, ρ.θ, ρ.n[1..ρ.K], ρ.N));
    } else {
      setWeight(observe_crp_categorical(x, ρ.α, ρ.θ, ρ.n[1..ρ.K], ρ.N));
    }
  }
}

/**
 * Create categorical distribution.
 */
function Categorical(ρ:RestaurantProcess) -> RestaurantProcessCategorical {
  x:RestaurantProcessCategorical;
  x.initialize(ρ);
  return x;
}
