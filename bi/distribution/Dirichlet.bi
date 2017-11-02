/**
 * Dirichlet distribution.
 *
 * `D` Number of components.
 */
class Dirichlet < Random<Real[_]> {
  /**
   * Concentration.
   */
  α:Real[_];

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
  m:Dirichlet;
  m.initialize(α);
  return m;
}

/**
 * Create Dirichlet distribution.
 */
function Dirichlet(α:Real, D:Integer) -> Dirichlet {
  return Dirichlet(vector(α, D));
}
