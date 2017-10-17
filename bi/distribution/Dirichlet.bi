/**
 * Dirichlet distribution.
 *
 * `D` Number of components.
 */
class Dirichlet(D:Integer) < DelayRealVector(D) {
  /**
   * Concentration.
   */
  α:Real[D];

  function initialize(α:Real[_]) {
    super.initialize();
    update(α);
  }

  function update(α:Real[_]) {
    this.α <- α;
  }

  function doRealize() {
    if (missing) {
      set(simulate_dirichlet(α));
    } else {
      setWeight(observe_dirichlet(x, α));
    }
  }

  function tildeLeft() -> Dirichlet {
    simulate();
    return this;
  }
}

/**
 * Create Dirichlet distribution.
 */
function Dirichlet(α:Real[_]) -> Dirichlet {
  m:Dirichlet(length(α));
  m.initialize(α);
  return m;
}
