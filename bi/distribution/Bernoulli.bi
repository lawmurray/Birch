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

  function graft() {
    if delay? {
      delay!.prune();
    } else {
      m:Beta?;
      if (m <- ρ.graftBeta())? {
        delay <- BetaBernoulli(future, futureUpdate, m!);
      } else {
        delay <- Bernoulli(future, futureUpdate, ρ);
      }
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
