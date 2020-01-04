/*
 * ed Poisson random variate.
 */
final class Poisson(future:Integer?, futureUpdate:Boolean,
    λ:Expression<Real>) < DelayMove<Discrete>(future, futureUpdate) {
  /**
   * Rate.
   */
  λ:Expression<Real> <- λ;

  function simulate() -> Integer {
    return simulate_poisson(λ.value());
  }

  function simulatePilot() -> Integer {
    return simulate_poisson(λ.pilot());
  }

  function simulatePropose() -> Integer {
    return simulate_poisson(λ.propose());
  }

  function logpdf(x:Integer) -> Real {
    return logpdf_poisson(x, λ.value());
  }

  function lazy(x:Expression<Integer>) -> Expression<Real> {
    return lazy_poisson(x, λ);
  }

  function cdf(x:Integer) -> Real? {
    return cdf_poisson(x, λ);
  }

  function quantile(p:Real) -> Integer? {
    return quantile_poisson(p, λ);
  }

  function lower() -> Integer? {
    return 0;
  }

  function graft() {
    if delay? {
      delay!.prune();
    } else {
      m1:TransformLinear<Gamma>?;
      m2:Gamma?;
      
      if (m1 <- λ.graftScaledGamma())? {
        delay <- ScaledGammaPoisson(future, futureUpdate, m1!.a, m1!.x);
      } else if (m2 <- λ.graftGamma())? {
        delay <- GammaPoisson(future, futureUpdate, m2!);
      } else {
        delay <- Poisson(future, futureUpdate, λ);
      }
    }
  }

  function graftDiscrete() -> Discrete? {
    graft();
    return Discrete?(delay);
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "Poisson");
    buffer.set("λ", λ.value());
  }
}

function Poisson(future:Integer?, futureUpdate:Boolean,
    λ:Expression<Real>) -> Poisson {
  o:Poisson(future, futureUpdate, λ.graft());
  return o;
}

/**
 * Create Poisson distribution.
 */
function Poisson(λ:Expression<Real>) -> Poisson {
  m:Poisson(λ);
  return m;
}

/**
 * Create Poisson distribution.
 */
function Poisson(λ:Real) -> Poisson {
  return Poisson(Boxed(λ));
}
