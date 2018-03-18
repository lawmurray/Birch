/**
 * Delta distribution, representing a distribution on a discrete space with
 * all probability mass at one location.
 */
class Delta < Random<Integer> {
  /**
   * Location.
   */
  μ:Integer;

  function initialize(μ:Integer) {
    super.initialize();
    update(μ);
  }

  function update(μ:Integer) {
    this.μ <- μ;
  }

  function doRealize() {
    if (isMissing()) {
      set(simulate_delta(μ));
    } else {
      setWeight(observe_delta(value(), μ));
    }
  }
}

/**
 * Create delta distribution.
 *
 * - μ: Location.
 */
function Delta(μ:Integer) -> Delta {
  m:Delta;
  m.initialize(μ);
  return m;
}
