/**
 * Dirichlet distribution.
 */
final class Dirichlet(α:Expression<Real[_]>) < Distribution<Real[_]> {
  /**
   * Concentration.
   */
  α:Expression<Real[_]> <- α;

  function valueForward() -> Real[_] {
    assert !delay?;
    return simulate_dirichlet(α);
  }

  function observeForward(x:Real[_]) -> Real {
    assert !delay?;
    return logpdf_dirichlet(x, α);
  }

  function graft(force:Boolean) {
    if delay? {
      delay!.prune();
    } else if force {
      delay <- DelayDirichlet(future, futureUpdate, α);
    }
  }

  function graftDirichlet() -> DelayDirichlet? {
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
