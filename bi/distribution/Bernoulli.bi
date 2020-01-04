/**
 * Bernoulli distribution.
 */
final class Bernoulli(future:Boolean?, futureUpdate:Boolean,
    ρ:Expression<Real>) < Distribution<Boolean>(future, futureUpdate) {
  /**
   * Success probability.
   */
  ρ:Expression<Real> <- ρ;

  function simulate() -> Boolean {
    return simulate_bernoulli(ρ);
  }
  
  function logpdf(x:Boolean) -> Real {
    return logpdf_bernoulli(x, ρ);
  }

  function graft() -> Distribution<Boolean> {
    prune();
    m:Beta?;
    if (m <- ρ.graftBeta())? {
      return BetaBernoulli(future, futureUpdate, m!);
    } else {
      return this;
    }
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
function Bernoulli(future:Boolean?, futureUpdate:Boolean,
    ρ:Expression<Real>) -> Bernoulli {
  m:Bernoulli(future, futureUpdate, ρ);
  return m;
}

/**
 * Create Bernoulli distribution.
 */
function Bernoulli(ρ:Expression<Real>) -> Bernoulli {
  return Bernoulli(nil, true, ρ);
}

/**
 * Create Bernoulli distribution.
 */
function Bernoulli(ρ:Real) -> Bernoulli {
  return Bernoulli(Boxed(ρ));
}
