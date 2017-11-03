/**
 * Categorical distribution.
 *
 * `D` Number of categories.
 */
class Categorical < Random<Integer> {
  /**
   * Category probabilities.
   */
  ρ:Real[_];

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
}

/**
 * Create categorical distribution.
 */
function Categorical(ρ:Real[_]) -> Categorical {
  m:Categorical;
  m.initialize(ρ);
  return m;
}
