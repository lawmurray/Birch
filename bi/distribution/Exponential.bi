/*
 * ed exponential random variate.
 */
final class Exponential(future:Real?, futureUpdate:Boolean, λ:Expression<Real>) <
    Distribution<Real>(future, futureUpdate) {
  /**
   * Rate.
   */
  λ:Expression<Real> <- λ;

  function simulate() -> Real {
    return simulate_exponential(λ);
  }

  function logpdf(x:Real) -> Real {
    return logpdf_exponential(x, λ);
  }

  function cdf(x:Real) -> Real? {
    return cdf_exponential(x, λ);
  }

  function quantile(p:Real) -> Real? {
    return quantile_exponential(p, λ);
  }

  function lower() -> Real? {
    return 0.0;
  }

  function graft() {
    if delay? {
      delay!.prune();
    } else {
      m1:TransformLinear<Gamma>?;
      m2:Gamma?;

      if (m1 <- λ.graftScaledGamma())? {
        delay <- ScaledGammaExponential(future, futureUpdate, m1!.a, m1!.x);
      } else if (m2 <- λ.graftGamma())? {
        delay <- GammaExponential(future, futureUpdate, m2!);
      } else {
        delay <- Exponential(future, futureUpdate, λ);
      }
    }
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "Exponential");
    buffer.set("λ", λ);
  }
}

function Exponential(future:Real?, futureUpdate:Boolean, λ:Real) ->
    Exponential {
  assert λ > 0.0;
  m:Exponential(future, futureUpdate, λ);
  return m;
}

/**
 * Create Exponential distribution.
 */
function Exponential(λ:Expression<Real>) -> Exponential {
  m:Exponential(λ);
  return m;
}

/**
 * Create Exponential distribution.
 */
function Exponential(λ:Real) -> Exponential {
  return Exponential(Boxed(λ));
}
