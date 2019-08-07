/**
 * Bernoulli distribution.
 */
final class Bernoulli(ρ:Expression<Real>) < Distribution<Boolean> {
  /**
   * Success probability.
   */
  ρ:Expression<Real> <- ρ;

  function valueForward() -> Boolean {
    assert !delay?;
    return simulate_bernoulli(ρ);
  }

  function observeForward(x:Boolean) -> Real {
    assert !delay?;
    return logpdf_bernoulli(x, ρ);
  }

  function graft(force:Boolean) {
    if delay? {
      delay!.prune();
    } else {
      m:DelayBeta?;
      if (m <- ρ.graftBeta())? {
        delay <- DelayBetaBernoulli(future, futureUpdate, m!);
      } else if force {
        delay <- DelayBernoulli(future, futureUpdate, ρ);
      }
    }
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
