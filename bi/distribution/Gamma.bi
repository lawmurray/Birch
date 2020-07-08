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

  function supportsLazy() -> Boolean {
    return true;
  }

  function simulate() -> Real {
    return simulate_gamma(k.value(), θ.value());
  }

  function simulateLazy() -> Real? {
    return simulate_gamma(k.get(), θ.get());
  }
  
  function logpdf(x:Real) -> Real {
    return logpdf_gamma(x, k.value(), θ.value());
  }

  function logpdfLazy(x:Expression<Real>) -> Expression<Real>? {
    return logpdf_lazy_gamma(x, k, θ);
  }

  function cdf(x:Real) -> Real? {
    return cdf_gamma(x, k.value(), θ.value());
  }

  function quantile(P:Real) -> Real? {
    return quantile_gamma(P, k.value(), θ.value());
  }

  function lower() -> Real? {
    return 0.0;
  }

  function graft() -> Distribution<Real> {
    prune();
    θ1:InverseGamma?;
    r:Distribution<Real> <- this;
    
    /* match a template */
    if (θ1 <- θ.graftInverseGamma())? {
      r <- InverseGammaGamma(k, θ1!);
    }

    return r;
  }

  function graftGamma() -> Gamma? {
    prune();
    return this;
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "Gamma");
    buffer.set("k", k);
    buffer.set("θ", θ);
  }
}

/**
 * Create gamma distribution.
 */
function Gamma(k:Expression<Real>, θ:Expression<Real>) -> Gamma {
  return construct<Gamma>(k, θ);
}

/**
 * Create gamma distribution.
 */
function Gamma(k:Expression<Real>, θ:Real) -> Gamma {
  return Gamma(k, box(θ));
}

/**
 * Create gamma distribution.
 */
function Gamma(k:Real, θ:Expression<Real>) -> Gamma {
  return Gamma(box(k), θ);
}

/**
 * Create gamma distribution.
 */
function Gamma(k:Real, θ:Real) -> Gamma {
  return Gamma(box(k), box(θ));
}
