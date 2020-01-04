/*
 * ed gamma random variate.
 */
final class Gamma(future:Real?, futureUpdate:Boolean, k:Expression<Real>, θ:Expression<Real>) <
    Distribution<Real>(future, futureUpdate) {
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

  function quantile(p:Real) -> Real? {
    return quantile_gamma(p, k, θ);
  }

  function lower() -> Real? {
    return 0.0;
  }

  function graft() {
    if delay? {
      delay!.prune();
    } else {
      θ1:InverseGamma?;
      if (θ1 <- θ.graftInverseGamma())? {
        delay <- InverseGammaGamma(future, futureUpdate, k, θ1!);
      } else {
        delay <- Gamma(future, futureUpdate, k, θ);
      }
    }
  }

  function graftGamma() -> Gamma? {
    if delay? {
      delay!.prune();
    } else {
      delay <- Gamma(future, futureUpdate, k, θ);
    }
    return Gamma?(delay);
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "Gamma");
    buffer.set("k", k);
    buffer.set("θ", θ);
  }
}

function Gamma(future:Real?, futureUpdate:Boolean, k:Real, θ:Real) ->
    Gamma {
  m:Gamma(future, futureUpdate, k, θ);
  return m;
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
