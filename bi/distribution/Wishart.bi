/**
 * Wishart distribution.
 */
final class Wishart(Ψ:Expression<LLT>, k:Expression<Real>) <
    Distribution<LLT> {
  /**
   * Scale.
   */
  Ψ:Expression<LLT> <- Ψ;
  
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

  function supportsLazy() -> Boolean {
    return true;
  }

  function simulate() -> LLT {
    return simulate_wishart(Ψ.value(), k.value());
  }

  function simulateLazy() -> LLT? {
    return simulate_wishart(Ψ.get(), k.get());
  }
  
  function logpdf(X:LLT) -> Real {
    return logpdf_wishart(X, Ψ.value(), k.value());
  }

  function logpdfLazy(X:Expression<LLT>) -> Expression<Real>? {
    return logpdf_lazy_wishart(X, Ψ, k);
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

/**
 * Create Wishart distribution.
 */
function Wishart(Ψ:Expression<LLT>, k:Expression<Real>) -> Wishart {
  m:Wishart(Ψ, k);
  return m;
}

/**
 * Create Wishart distribution.
 */
function Wishart(Ψ:Expression<LLT>, k:Real) -> Wishart {
  return Wishart(Ψ, box(k));
}

/**
 * Create Wishart distribution.
 */
function Wishart(Ψ:LLT, k:Expression<Real>) -> Wishart {
  return Wishart(box(Ψ), k);
}

/**
 * Create Wishart distribution.
 */
function Wishart(Ψ:LLT, k:Real) -> Wishart {
  return Wishart(box(Ψ), box(k));
}

/**
 * Create Wishart distribution.
 */
function Wishart(Ψ:Expression<Real[_,_]>, k:Expression<Real>) -> Wishart {
  return Wishart(llt(Ψ), k);
}

/**
 * Create Wishart distribution.
 */
function Wishart(Ψ:Expression<Real[_,_]>, k:Real) -> Wishart {
  return Wishart(llt(Ψ), k);
}

/**
 * Create Wishart distribution.
 */
function Wishart(Ψ:Real[_,_], k:Expression<Real>) -> Wishart {
  return Wishart(llt(Ψ), k);
}

/**
 * Create Wishart distribution.
 */
function Wishart(Ψ:Real[_,_], k:Real) -> Wishart {
  return Wishart(llt(Ψ), k);
}
