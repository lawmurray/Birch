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

  function graft() {
    if delay? {
      delay!.prune();
    } else {
      θ1:DelayInverseGamma?;
      if (θ1 <- θ.graftInverseGamma())? {
        delay <- DelayInverseGammaGamma(future, futureUpdate, k, θ1!);
      } else {
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

  function write(buffer:Buffer) {
    if delay? {
      delay!.write(buffer);
    } else {
      buffer.set("class", "Gamma");
      buffer.set("k", k.value());
      buffer.set("θ", θ.value());
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
