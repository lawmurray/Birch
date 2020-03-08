/*
 * ed Dirichlet random variate.
 */
final class Dirichlet(α:Expression<Real[_]>) < Distribution<Real[_]> {
  /**
   * Concentration.
   */
  α:Expression<Real[_]> <- α;

  function simulate() -> Real[_] {
    return simulate_dirichlet(α.value());
  }
  
  function logpdf(x:Real[_]) -> Real {
    return logpdf_dirichlet(x, α.value());
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
