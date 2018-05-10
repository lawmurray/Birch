/**
 * Gamma distribution.
 */
class Gamma(k:Expression<Real>, θ:Expression<Real>) < Distribution<Real> {
  /**
   * Shape.
   */
  k:Expression<Real> <- k;
  
  /**
   * Scale.
   */
  θ:Expression<Real> <- θ;

  function graft() {
    if delay? {
      delay!.prune();
    } else {
      delay <- DelayGamma(x, k, θ);
    }
  }

  function graftGamma() -> DelayGamma? {
    if delay? {
      delay!.prune();
    } else {
      delay <- DelayGamma(x, k, θ);
    }
    return DelayGamma?(delay);
  }
}

/**
 * Create gamma distribution.
 */
function Gamma(k:Expression<Real>, θ:Expression<Real>) -> Gamma {
  m:Gamma(k, θ);
  return m;
}

/**
 * Create gamma distribution.
 */
function Gamma(k:Expression<Real>, θ:Real) -> Gamma {
  return Gamma(k, Boxed(θ));
}

/**
 * Create gamma distribution.
 */
function Gamma(k:Real, θ:Expression<Real>) -> Gamma {
  return Gamma(Boxed(k), θ);
}

/**
 * Create gamma distribution.
 */
function Gamma(k:Real, θ:Real) -> Gamma {
  return Gamma(Boxed(k), Boxed(θ));
}
