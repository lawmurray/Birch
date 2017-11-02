/**
 * Multinomial distribution.
 *
 * `D` Number of categories.
 */
class Multinomial < Random<Integer[_]> {
  /**
   * Number of trials.
   */
  n:Integer;

  /**
   * Category probabilities.
   */
  ρ:Real[_];

  function initialize(n:Integer, ρ:Real[_]) {
    super.initialize();
    update(n, ρ);
  }

  function update(n:Integer, ρ:Real[_]) {
    this.n <- n;
    this.ρ <- ρ;
  }

  function doRealize() {
    if (missing) {
      set(simulate_multinomial(n, ρ));
    } else {
      setWeight(observe_multinomial(x, ρ));
    }
  }

  function tildeLeft() -> Multinomial {
    simulate();
    return this;
  }
}

/**
 * Create multinomial distribution.
 */
function Multinomial(n:Integer, ρ:Real[_]) -> Multinomial {
  m:Multinomial;
  m.initialize(n, ρ);
  return m;
}
