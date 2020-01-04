/*
 * ed Dirichlet random variate.
 */
final class Dirichlet(future:Real[_]?, futureUpdate:Boolean, α:Expression<Real[_]>)
    < Distribution<Real[_]>(future, futureUpdate) {
  /**
   * Concentration.
   */
  α:Expression<Real[_]> <- α;

  function simulate() -> Real[_] {
    return simulate_dirichlet(α);
  }
  
  function logpdf(x:Real[_]) -> Real {
    return logpdf_dirichlet(x, α);
  }

  function graft() -> Distribution<Real[_]> {
    prune();
    return this;
  }

  function graftDirichlet() -> Dirichlet? {
    prune();
    return this;
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "Dirichlet");
    buffer.set("α", α);
  }
}

function Dirichlet(future:Real[_]?, futureUpdate:Boolean,
    α:Expression<Real[_]>) -> Dirichlet {
  m:Dirichlet(future, futureUpdate, α);
  return m;
}

/**
 * Create Dirichlet distribution.
 */
function Dirichlet(α:Expression<Real[_]>) -> Dirichlet {
  m:Dirichlet(nil, true, α);
  return m;
}

/**
 * Create Dirichlet distribution.
 */
function Dirichlet(α:Real[_]) -> Dirichlet {
  return Dirichlet(Boxed(α));
}
