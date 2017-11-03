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
    this.α <- α;
  }

  /**
   * Update with draw from multinomial distribution.
   */
  function update(x:Integer[_]) {
    assert length(x) == length(α);
    for (i:Integer in 1..length(x)) {
      α[i] <- α[i] + x[i];
    }
  }

  /**
   * Update with draw from categorical distribution.
   */
  function update(x:Integer) {
    α[x] <- α[x] + 1.0;
  }

  function doRealize() {
    if (missing) {
      set(simulate_dirichlet(α));
    } else {
      setWeight(observe_dirichlet(x, α));
    }
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
