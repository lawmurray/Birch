/**
 * Categorical distribution.
 *
 * `D` Number of categories.
 */
class Categorical(D:Integer) < DelayInteger {
  /**
   * Category probabilities.
   */
  ρ:Real[D];

  function initialize(ρ:Real[_]) {
    super.initialize();
    update(ρ);
  }

  function update(ρ:Real[_]) {
    this.ρ <- ρ;
  }

  function doRealize() {
    if (missing) {
      set(simulate_categorical(ρ));
    } else {
      setWeight(observe_categorical(x, ρ));
    }
  }

  function tildeLeft() -> Categorical {
    simulate();
    return this;
  }
}

/**
 * Create categorical distribution.
 */
function Categorical(ρ:Real[_]) -> Categorical {
  m:Categorical(length(ρ));
  m.initialize(ρ);
  return m;
}
