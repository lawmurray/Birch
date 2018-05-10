/**
 * Bernoulli distribution.
 */
class Bernoulli(ρ:Expression<Real>) < Distribution<Boolean> {
  /**
   * Success probability.
   */
  ρ:Expression<Real> <- ρ;

  function graft() {
    if delay? {
      delay!.prune();
    } else {
      m:DelayBeta?;
      if (m <- ρ.graftBeta())? {
        delay <- DelayBetaBernoulli(x, m!);
      } else {
        delay <- DelayBernoulli(x, ρ);
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
