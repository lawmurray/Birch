/**
 * Poisson distribution.
 */
class Poisson < DelayInteger {
  /**
   * Rate.
   */
  λ:Real;

  function initialize(λ:Real) {
    super.initialize();
    update(λ);
  }

  function update(λ:Real) {
    assert 0.0 < λ;
  
    this.λ <- λ;
  }

  function doRealize() {
    if (isMissing()) {
      set(simulate_poisson(λ));
    } else {
      setWeight(observe_poisson(x, λ));
    }
  }

  function tildeLeft() -> Poisson {
    simulate();
    return this;
  }
  
  function tildeRight(left:Poisson) -> Poisson {
    set(left.value());
    observe();
    return this;
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
