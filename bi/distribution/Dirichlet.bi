/**
 * Dirichlet distribution.
 */
class Dirichlet<Type1>(α:Type1) < Random<Real[_]> {
  /**
   * Concentration.
   */
  α:Type1;

  /**
   * Update with draw from multinomial distribution.
   */
  function update(x:Integer[_]) {
    //α <- α + x;
  }

  /**
   * Update with draw from categorical distribution.
   */
  function update(x:Integer) {
    //α[x] <- α[x] + 1.0;
  }

  function doSimulate() -> Real[_] {
    return simulate_dirichlet(global.value(α));
  }
  
  function doObserve(x:Real[_]) -> Real {
    return observe_dirichlet(x, global.value(α));
  }
}

/**
 * Create Dirichlet distribution.
 */
function Dirichlet(α:Real[_]) -> Dirichlet<Real[_]> {
  m:Dirichlet<Real[_]>(α);
  m.initialize();
  return m;
}

/**
 * Create Dirichlet distribution.
 */
function Dirichlet(α:Expression<Real[_]>) -> Dirichlet<Expression<Real[_]>> {
  m:Dirichlet<Expression<Real[_]>>(α);
  m.initialize();
  return m;
}

/**
 * Create Dirichlet distribution.
 */
function Dirichlet(α:Real, D:Integer) -> Dirichlet<Real[_]> {
  return Dirichlet(vector(α, D));
}
