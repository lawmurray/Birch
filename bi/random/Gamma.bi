/**
 * Gamma distribution.
 */
class Gamma(k:Expression<Real>, θ:Expression<Real>) < Random<Real> {
  /**
   * Shape.
   */
  k:Expression<Real> <- k;
  
  /**
   * Scale.
   */
  θ:Expression<Real> <- θ;

  function graft() -> Delay? {
    if (delay?) {
      return delay;
    } else {
      return DelayGamma(this, k, θ);
    }
  }

  function graftGamma() -> DelayGamma? {
    if (delay?) {
      return DelayGamma?(delay);
    } else {
      return DelayGamma(this, k, θ);
    }
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
