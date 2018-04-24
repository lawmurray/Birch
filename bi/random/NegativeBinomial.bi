/**
 * Negative binomial distribution.
 */
class NegativeBinomial(k:Expression<Integer>, ρ:Expression<Real>) < Random<Integer> {
  /**
   * Number of successes before the experiment is stopped.
   */
  k:Expression<Integer> <- k;

  /**
   * Probability of success.
   */
  ρ:Expression<Real> <- ρ;

  function doSimulate() -> Integer {
    return simulate_negative_binomial(k.value(), ρ.value());
  }
  
  function doObserve(x:Integer) -> Real {
    return observe_negative_binomial(x, k.value(), ρ.value());
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
  return NegativeBinomial(k, Literal(ρ));
}

/**
 * Create negative binomial distribution.
 */
function NegativeBinomial(k:Integer, ρ:Expression<Real>) -> NegativeBinomial {
  return NegativeBinomial(Literal(k), ρ);
}

/**
 * Create negative binomial distribution.
 */
function NegativeBinomial(k:Integer, ρ:Real) -> NegativeBinomial {
  return NegativeBinomial(Literal(k), Literal(ρ));
}
