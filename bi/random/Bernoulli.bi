/**
 * Bernoulli distribution.
 */
class Bernoulli(ρ:Expression<Real>) < Random<Boolean> {
  /**
   * Success probability.
   */
  ρ:Expression<Real> <- ρ;

  function graft() {
    if (ρ.isBeta()) {
      m:DelayBetaBernoulli(this, ρ.getBeta());
      m.graft();
      delay <- m;
    } else {
      m:DelayBernoulli(this, ρ.value());
      m.graft();
      delay <- m;
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
