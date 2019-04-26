/**
 * Dirichlet distribution.
 */
final class Dirichlet(α:Expression<Real[_]>) < Distribution<Real[_]> {
  /**
   * Concentration.
   */
  α:Expression<Real[_]> <- α;

  function graft() {
    if delay? {
      delay!.prune();
    } else {
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

  function write(buffer:Buffer) {
    if delay? {
      delay!.write(buffer);
    } else {
      buffer.set("class", "Dirichlet");
      buffer.set("α", α.value());
    }
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
