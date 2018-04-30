/**
 * Bernoulli distribution.
 */
class Bernoulli(ρ:Expression<Real>) < Random<Boolean> {
  /**
   * Success probability.
   */
  ρ:Expression<Real> <- ρ;

  function doGraft() -> DelayValue<Boolean>? {
    m:DelayBeta?;
    if (m <- ρ.graftBeta())? {
      return DelayBetaBernoulli(this, m!);
    } else {
      return DelayBernoulli(this, ρ.value());
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
