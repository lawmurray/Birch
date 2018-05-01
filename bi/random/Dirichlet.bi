/**
 * Dirichlet distribution.
 */
class Dirichlet(α:Expression<Real[_]>) < Random<Real[_]> {
  /**
   * Concentration.
   */
  α:Expression<Real[_]> <- α;

  function doGraft() -> DelayValue<Real[_]>? {
    return DelayDirichlet(this, α);
  }

  function doGraftDirichlet() -> DelayDirichlet? {
    return DelayDirichlet(this, α);
  }
}

/**
 * Create Dirichlet distribution.
 */
function Dirichlet(α:Expression<Real[_]>) -> Dirichlet {
  m:Dirichlet(α);
  return m;
}

/**
 * Create Dirichlet distribution.
 */
function Dirichlet(α:Real[_]) -> Dirichlet {
  return Dirichlet(Boxed(α));
}
