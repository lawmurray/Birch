/**
 * Dirichlet distribution.
 */
final class Dirichlet(α:Expression<Real[_]>) < Distribution<Real[_]> {
  /**
   * Concentration.
   */
  α:Expression<Real[_]> <- α;

  function graft(child:Delay?) {
    if delay? {
      delay!.prune();
    } else {
      delay <- DelayDirichlet(future, futureUpdate, α);
    }
  }

  function graftDirichlet(child:Delay?) -> DelayDirichlet? {
    if delay? {
      delay!.prune();
    } else {
      delay <- DelayDirichlet(future, futureUpdate, α);
    }
    return DelayDirichlet?(delay);
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
