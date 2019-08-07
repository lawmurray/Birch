/**
 * Gamma distribution.
 */
final class Gamma(k:Expression<Real>, θ:Expression<Real>) < Distribution<Real> {
  /**
   * Shape.
   */
  k:Expression<Real> <- k;
  
  /**
   * Scale.
   */
  θ:Expression<Real> <- θ;

  function valueForward() -> Real {
    assert !delay?;
    return simulate_gamma(k, θ);
  }

  function observeForward(x:Real) -> Real {
    assert !delay?;
    return logpdf_gamma(x, k, θ);
  }

  function graft(force:Boolean) {
    if delay? {
      delay!.prune();
    } else {
      θ1:DelayInverseGamma?;
      if (θ1 <- θ.graftInverseGamma())? {
        delay <- DelayInverseGammaGamma(future, futureUpdate, k, θ1!);
      } else if force {
        delay <- DelayGamma(future, futureUpdate, k, θ);
      }
    }
  }

  function graftGamma() -> DelayGamma? {
    if delay? {
      delay!.prune();
    } else {
      delay <- DelayGamma(future, futureUpdate, k, θ);
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
