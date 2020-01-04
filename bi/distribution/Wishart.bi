/*
 * ed Wishart random variate.
 */
final class Wishart(future:Real[_,_]?, futureUpdate:Boolean,
    Ψ:Expression<Real[_,_]>, k:Expression<Real>) <
    Distribution<Real[_,_]>(future, futureUpdate) {
  /**
   * Scale.
   */
  Ψ:Expression<Real[_,_]> <- Ψ;
  
  /**
   * Degrees of freedom.
   */
  k:Expression<Real> <- k;

  function rows() -> Integer {
    return Ψ.rows();
  }

  function columns() -> Integer {
    return Ψ.columns();
  }

  function simulate() -> Real[_,_] {
    return simulate_wishart(Ψ, k);
  }
  
  function logpdf(X:Real[_,_]) -> Real {
    return logpdf_wishart(X, Ψ, k);
  }

  function graft() -> Distribution<Real[_,_]> {
    prune();
    return this;
  }

  function graftWishart() -> Wishart? {
    prune();
    return this;
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "Wishart");
    buffer.set("Ψ", Ψ);
    buffer.set("k", k);
  }
}

function Wishart(future:Real[_,_]?, futureUpdate:Boolean,
    Ψ:Expression<Real[_,_]>, k:Expression<Real>) -> Wishart {
  m:Wishart(future, futureUpdate, Ψ, k);
  return m;
}

/**
 * Create Wishart distribution.
 */
function Wishart(Ψ:Expression<Real[_,_]>, k:Expression<Real>) -> Wishart {
  m:Wishart(nil, true, Ψ, k);
  return m;
}

/**
 * Create Wishart distribution.
 */
function Wishart(Ψ:Expression<Real[_,_]>, k:Real) -> Wishart {
  return Wishart(Ψ, Boxed(k));
}

/**
 * Create Wishart distribution.
 */
function Wishart(Ψ:Real[_,_], k:Expression<Real>) -> Wishart {
  return Wishart(Boxed(Ψ), k);
}

/**
 * Create Wishart distribution.
 */
function Wishart(Ψ:Real[_,_], k:Real) -> Wishart {
  return Wishart(Boxed(Ψ), Boxed(k));
}
