/**
 * Bernoulli distribution.
 */
class Bernoulli < DelayBoolean {
  /**
   * Probability of a true result.
   */
  ρ:Real;

  function initialize(ρ:Real) {
    super.initialize();
    update(ρ);
  }

  function update(ρ:Real) {
    assert 0.0 <= ρ && ρ <= 1.0;
  
    this.ρ <- ρ;
  }

  function doRealize() {
    if (isMissing()) {
      set(simulate_bernoulli(ρ));
    } else {
      setWeight(observe_bernoulli(x, ρ));
    }
  }

  function tildeLeft() -> Bernoulli {
    simulate();
    return this;
  }
}

/**
 * Create Bernoulli distribution.
 */
function Bernoulli(ρ:Real) -> Bernoulli {
  m:Bernoulli;
  m.initialize(ρ);
  return m;
}
