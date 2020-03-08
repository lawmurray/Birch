/*
 * ed exponential random variate.
 */
final class Exponential(λ:Expression<Real>) <
    Distribution<Real> {
  /**
   * Rate.
   */
  λ:Expression<Real> <- λ;

  function simulate() -> Real {
    return simulate_exponential(λ.value());
  }

  function logpdf(x:Real) -> Real {
    return logpdf_exponential(x, λ.value());
  }

  function cdf(x:Real) -> Real? {
    return cdf_exponential(x, λ.value());
  }

  function quantile(P:Real) -> Real? {
    return quantile_exponential(P, λ.value());
  }

  function lower() -> Real? {
    return 0.0;
  }

  function graft() -> Distribution<Real> {
    prune();
    m1:TransformLinear<Gamma>?;
    m2:Gamma?;
    if (m1 <- λ.graftScaledGamma())? {
      return ScaledGammaExponential(m1!.a, m1!.x);
    } else if (m2 <- λ.graftGamma())? {
      return GammaExponential(m2!);
    } else {
      return this;
    }
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "Exponential");
    buffer.set("λ", λ);
  }
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
