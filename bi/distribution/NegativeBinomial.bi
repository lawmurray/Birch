/**
 * Negative binomial distribution.
 */
class NegativeBinomial(k:Expression<Integer>, ρ:Expression<Real>) <
    Distribution<Integer> {
  /**
   * Number of successes before the experiment is stopped.
   */
  k:Expression<Integer> <- k;

  /**
   * Success probability.
   */
  ρ:Expression<Real> <- ρ;

  function graft() {
    if delay? {
      delay!.prune();
    } else {
      ρ1:DelayBeta?; // Node for the prior on the probability

      if (ρ1 <- ρ.graftBeta())? {
        // If the probability is Beta, graft a Beta-Negative-Binomial
        delay <- DelayBetaNegativeBinomial(x, k, ρ1!);
      } else {
        delay <- DelayNegativeBinomial(x, k, ρ);
      }
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
