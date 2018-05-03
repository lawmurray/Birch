/**
 * Bernoulli distribution.
 */
class Bernoulli(ρ:Expression<Real>) < Random<Boolean> {
  /**
   * Success probability.
   */
  ρ:Expression<Real> <- ρ;

  function graft() -> Delay? {
    if (delay?) {
      return delay;
    } else {
      m:DelayBeta?;
      if (m <- ρ.graftBeta())? {
        return DelayBetaBernoulli(this, m!);
      } else {
        return DelayBernoulli(this, ρ);
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
