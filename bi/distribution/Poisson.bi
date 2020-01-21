/*
 * ed Poisson random variate.
 */
final class Poisson(λ:Expression<Real>) < Discrete {
  /**
   * Rate.
   */
  λ:Expression<Real> <- λ;

  function simulate() -> Integer {
    return simulate_poisson(λ);
  }

  function logpdf(x:Integer) -> Real {
    return logpdf_poisson(x, λ);
  }

  function cdf(x:Integer) -> Real? {
    return cdf_poisson(x, λ);
  }

  function quantile(P:Real) -> Integer? {
    return quantile_poisson(P, λ);
  }

  function lower() -> Integer? {
    return 0;
  }

  function logpdfLazy(x:Expression<Integer>) -> Expression<Real>? {
    return logpdf_lazy_poisson(x, λ);
  }

  function graft() -> Distribution<Integer> {
    prune();
    m1:TransformLinear<Gamma>?;
    m2:Gamma?;
      
    if (m1 <- λ.graftScaledGamma())? {
      return ScaledGammaPoisson(m1!.a, m1!.x);
    } else if (m2 <- λ.graftGamma())? {
      return GammaPoisson(m2!);
    } else {
      return this;
    }
  }

  function graftDiscrete() -> Discrete? {
    prune();
    return this;
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "Poisson");
    buffer.set("λ", λ.value());
  }
}

/**
 * Create Poisson distribution.
 */
function Poisson(λ:Expression<Real>) -> Poisson {
  o:Poisson(λ);
  return o;
}

/**
 * Create Poisson distribution.
 */
function Poisson(λ:Real) -> Poisson {
  return Poisson(Boxed(λ));
}
