/**
 * Binomial distribution.
 */
class Binomial(n:Expression<Integer>, ρ:Expression<Real>) < Distribution<Integer> {
  /**
   * Number of trials.
   */
  n:Expression<Integer> <- n;

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
        delay <- DelayBetaBinomial(x, n, m!);
      } else {
        delay <- DelayBinomial(x, n, ρ);
      }
    }
  }
  
  function graftDiscrete() -> DelayDiscrete? {
    if delay? {
      delay!.prune();
    } else {
      delay <- DelayBinomial(x, n, ρ);
    }
    return DelayValue<Integer>?(delay);
  }

  function graftBoundedDiscrete() -> DelayBoundedDiscrete? {
    if delay? {
      delay!.prune();
    } else {
      delay <- DelayBinomial(x, n, ρ);
    }
    return DelayBoundedDiscrete?(delay);
  }
}

/**
 * Create binomial distribution.
 */
function Binomial(n:Expression<Integer>, ρ:Expression<Real>) -> Binomial {
  m:Binomial(n, ρ);
  return m;
}

/**
 * Create binomial distribution.
 */
function Binomial(n:Expression<Integer>, ρ:Real) -> Binomial {
  return Binomial(n, Boxed(ρ));
}

/**
 * Create binomial distribution.
 */
function Binomial(n:Integer, ρ:Expression<Real>) -> Binomial {
  return Binomial(Boxed(n), ρ);
}

/**
 * Create binomial distribution.
 */
function Binomial(n:Integer, ρ:Real) -> Binomial {
  return Binomial(Boxed(n), Boxed(ρ));
}
