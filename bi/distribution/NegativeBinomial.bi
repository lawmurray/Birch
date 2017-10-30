/**
 * Negative binomial distribution.
 */
class NegativeBinomial < Random<Integer> {
  /**
   * Number of successes before the experiment is stopped.
   */
  k:Integer;

  /**
   * Probability of success.
   */
  ρ:Real;

  function initialize(k:Integer, ρ:Real) {
    super.initialize();
    update(k, ρ);
  }

  function update(k:Integer, ρ:Real) {
    assert 0 < k;
    assert 0.0 <= ρ && ρ <= 1.0;
  
    this.k <- k;
    this.ρ <- ρ;
  }

  function doRealize() {
    if (isMissing()) {
      set(simulate_negative_binomial(k, ρ));
    } else {
      setWeight(observe_negative_binomial(x, k, ρ));
    }
  }

  function tildeLeft() -> NegativeBinomial {
    simulate();
    return this;
  }
}

/**
 * Create.
 */
function NegativeBinomial(k:Integer, ρ:Real) -> NegativeBinomial {
  m:NegativeBinomial;
  m.initialize(k, ρ);
  return m;
}
