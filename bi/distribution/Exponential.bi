/**
 * Exponential distribution.
 */
class Exponential(λ:Expression<Real>) < Distribution<Real> {
  /**
   * Rate.
   */
  λ:Expression<Real> <- λ;

  function supportsLazy() -> Boolean {
    return true;
  }

  function simulate() -> Real {
    return simulate_exponential(λ.value());
  }

  function simulateLazy() -> Real? {
    return simulate_exponential(λ.get());
  }

  function logpdf(x:Real) -> Real {
    return logpdf_exponential(x, λ.value());
  }

  function logpdfLazy(x:Expression<Real>) -> Expression<Real>? {
    return logpdf_lazy_exponential(x, λ);
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
    r:Distribution<Real> <- this;
    
    /* match a template */
    if (m1 <- λ.graftScaledGamma())? {
      r <- ScaledGammaExponential(m1!.a, m1!.x);
    } else if (m2 <- λ.graftGamma())? {
      r <- GammaExponential(m2!);
    }

    return r;
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
  return construct<Exponential>(λ);
}

/**
 * Create Exponential distribution.
 */
function Exponential(λ:Real) -> Exponential {
  return Exponential(box(λ));
}
