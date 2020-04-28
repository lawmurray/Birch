/**
 * Bernoulli distribution.
 */
class Bernoulli(ρ:Expression<Real>) < Distribution<Boolean> {
  /**
   * Success probability.
   */
  ρ:Expression<Real> <- ρ;

  function simulate() -> Boolean {
    return simulate_bernoulli(ρ.value());
  }
  
  function logpdf(x:Boolean) -> Real {
    return logpdf_bernoulli(x, ρ.value());
  }

  function graft() -> Distribution<Boolean> {
    prune();
    m:Beta?;
    r:Distribution<Boolean> <- this;
    
    /* match a template */
    if (m <- ρ.graftBeta())? {
      r <- BetaBernoulli(m!);
    }

    return r;
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "Bernoulli");
    buffer.set("ρ", ρ);
  }
}

/**
 * Create Bernoulli distribution.
 */
function Bernoulli(ρ:Expression<Real>) -> Bernoulli {
  m:Bernoulli(ρ);
  return m;
}

/**
 * Create Bernoulli distribution.
 */
function Bernoulli(ρ:Real) -> Bernoulli {
  return Bernoulli(Boxed(ρ));
}
