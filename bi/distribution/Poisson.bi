/**
 * Poisson distribution.
 */
class Poisson < Random<Integer> {
  /**
   * Rate.
   */
  λ:Real;

  function initialize(λ:Real) {
    super.initialize();
    update(λ);
  }

  function update(λ:Real) {
    assert 0.0 <= λ;
  
    this.λ <- λ;
  }

  function doRealize() {
    if (isMissing()) {
      set(simulate_poisson(λ));
    } else {
      setWeight(observe_poisson(value(), λ));
    }
  }
}

/**
 * Create Poisson distribution.
 */
function Poisson(λ:Real) -> Poisson {
  m:Poisson;
  m.initialize(λ);
  return m;
}
