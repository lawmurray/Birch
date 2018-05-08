/**
 * Bernoulli distribution.
 */
class Bernoulli(ρ:Expression<Real>) < Distribution<Boolean> {
  /**
   * Success probability.
   */
  ρ:Expression<Real> <- ρ;

  function graft() -> DelayValue<Boolean> {
    m:DelayBeta?;
    if (m <- ρ.graftBeta())? {
      return DelayBetaBernoulli(m!);
    } else {
      return DelayBernoulli(ρ);
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
