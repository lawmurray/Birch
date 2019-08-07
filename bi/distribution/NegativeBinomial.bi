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

  function valueForward() -> Integer {
    assert !delay?;
    return simulate_negative_binomial(k, ρ);
  }

  function observeForward(x:Integer) -> Real {
    assert !delay?;
    return logpdf_negative_binomial(x, k, ρ);
  }

  function graft(force:Boolean) {
    if delay? {
      delay!.prune();
    } else {
      ρ1:DelayBeta?;
      if (ρ1 <- ρ.graftBeta())? {
        delay <- DelayBetaNegativeBinomial(future, futureUpdate, k, ρ1!);
      } else if force {
        delay <- DelayNegativeBinomial(future, futureUpdate, k, ρ);
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
