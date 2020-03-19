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

  function simulate() -> Real {
    return simulate_gamma(k.value(), θ.value());
  }
  
  function logpdf(x:Real) -> Real {
    return logpdf_gamma(x, k.value(), θ.value());
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
    if !hasValue() {
      prune();
      θ1:InverseGamma?;
      r:Distribution<Real>?;
    
      /* match a template */
      if (θ1 <- θ.graftInverseGamma())? {
        r <- InverseGammaGamma(k, θ1!);
      }

      /* finalize, and if not valid, use default template */
      if !r? || !r!.graftFinalize() {
        r <- GraftedGamma(k, θ);
        r!.graftFinalize();
      }
      return r!;
    } else {
      return this;
    }
  }

  function graftGamma() -> Gamma? {
    if !hasValue() {
      prune();
      auto r <- GraftedGamma(k, θ);
      r.graftFinalize();
      return r;
    } else {
      return nil;
    }
  }

  function graftFinalize() -> Boolean {
    assert false;  // should have been replaced during graft
    return false;
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
