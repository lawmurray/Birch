/**
 * Negative binomial distribution.
 */
class NegativeBinomial(k:Expression<Integer>, ρ:Expression<Real>) <
    Random<Integer> {
  /**
   * Number of successes before the experiment is stopped.
   */
  k:Expression<Integer> <- k;

  /**
   * Success probability.
   */
  ρ:Expression<Real> <- ρ;

  function graft() -> Delay? {
    if (delay?) {
      return delay;
    } else {
      return DelayNegativeBinomial(this, k.value(), ρ.value());
    }
  }
}

/**
 * Create negative binomial distribution.
 */
function NegativeBinomial(k:Expression<Integer>, ρ:Expression<Real>) -> NegativeBinomial {
  m:NegativeBinomial(k, ρ);
  return m;
}

/**
 * Create negative binomial distribution.
 */
function NegativeBinomial(k:Expression<Integer>, ρ:Real) -> NegativeBinomial {
  return NegativeBinomial(k, Boxed(ρ));
}

/**
 * Create negative binomial distribution.
 */
function NegativeBinomial(k:Integer, ρ:Expression<Real>) -> NegativeBinomial {
  return NegativeBinomial(Boxed(k), ρ);
}

/**
 * Create negative binomial distribution.
 */
function NegativeBinomial(k:Integer, ρ:Real) -> NegativeBinomial {
  return NegativeBinomial(Boxed(k), Boxed(ρ));
}
