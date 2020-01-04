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

  function graft() {
    if delay? {
      delay!.prune();
    } else {
      delay <- Dirichlet(future, futureUpdate, α);
    }
  }

  function graftDirichlet() -> Dirichlet? {
    if delay? {
      delay!.prune();
    } else {
      delay <- Dirichlet(future, futureUpdate, α);
    }
    return Dirichlet?(delay);
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "Dirichlet");
    buffer.set("α", α);
  }
}

function Dirichlet(future:Real[_]?, futureUpdate:Boolean, α:Real[_]) ->
    Dirichlet {
  m:Dirichlet(future, futureUpdate, α);
  return m;
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
