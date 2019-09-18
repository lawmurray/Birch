/**
 * Wishart distribution.
 */
final class Wishart(Ψ:Expression<Real[_,_]>, ν:Expression<Real>) <
    Distribution<Real[_,_]> {
  /**
   * Scale.
   */
  Ψ:Expression<Real[_,_]> <- Ψ;
  
  /**
   * Degrees of freedom.
   */
  ν:Expression<Real> <- ν;

  function valueForward() -> Real[_,_] {
    assert !delay?;
    return simulate_wishart(Ψ, ν);
  }

  function observeForward(X:Real[_,_]) -> Real {
    assert !delay?;
    return logpdf_wishart(X, Ψ, ν);
  }

  function graft(force:Boolean) {
    if delay? {
      delay!.prune();
    } else if force {
      delay <- DelayWishart(future, futureUpdate, Ψ, ν);
    }
  }

  function graftWishart() -> DelayWishart? {
    if delay? {
      delay!.prune();
    } else {
      delay <- DelayWishart(future, futureUpdate, Ψ, ν);
    }
    return DelayWishart?(delay);
  }
}

/**
 * Create Wishart distribution.
 */
function Wishart(Ψ:Expression<Real[_,_]>, ν:Expression<Real>) -> Wishart {
  m:Wishart(Ψ, ν);
  return m;
}

/**
 * Create Wishart distribution.
 */
function Wishart(Ψ:Expression<Real[_,_]>, ν:Real) -> Wishart {
  return Wishart(Ψ, Boxed(ν));
}

/**
 * Create Wishart distribution.
 */
function Wishart(Ψ:Real[_,_], ν:Expression<Real>) -> Wishart {
  return Wishart(Boxed(Ψ), ν);
}

/**
 * Create Wishart distribution.
 */
function Wishart(Ψ:Real[_,_], ν:Real) -> Wishart {
  return Wishart(Boxed(Ψ), Boxed(ν));
}
