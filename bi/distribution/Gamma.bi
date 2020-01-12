/*
 * ed gamma random variate.
 */
final class Gamma(k:Expression<Real>, θ:Expression<Real>) <
    Distribution<Real> {
  /**
   * Shape.
   */
  k:Expression<Real> <- k;
  
  /**
   * Scale.
   */
  θ:Expression<Real> <- θ;

  function simulate() -> Real {
    return simulate_gamma(k, θ);
  }
  
  function logpdf(x:Real) -> Real {
    return logpdf_gamma(x, k, θ);
  }

  function cdf(x:Real) -> Real? {
    return cdf_gamma(x, k, θ);
  }

  function quantile(P:Real) -> Real? {
    return quantile_gamma(P, k, θ);
  }

  function lower() -> Real? {
    return 0.0;
  }

  function graft() -> Distribution<Real> {
    prune();
    θ1:InverseGamma?;
    if (θ1 <- θ.graftInverseGamma())? {
      return InverseGammaGamma(k, θ1!);
    } else {
      return this;
    }
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
