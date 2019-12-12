/**
 * Binomial distribution.
 */
final class Binomial(n:Expression<Integer>, ρ:Expression<Real>) < Distribution<Integer> {
  /**
   * Number of trials.
   */
  n:Expression<Integer> <- n;

  /**
   * Success probability.
   */
  ρ:Expression<Real> <- ρ;

  function graft(child:Delay?) {
    if delay? {
      delay!.prune();
    } else {
      m:DelayBeta?;
      if (m <- ρ.graftBeta(child))? {
        delay <- DelayBetaBinomial(future, futureUpdate, n, m!);
      } else {
        delay <- DelayBinomial(future, futureUpdate, n, ρ);
      }
    }
  }
  
  function graftDiscrete(child:Delay?) -> DelayDiscrete? {
    return graftBoundedDiscrete(child);
  }

  function graftBoundedDiscrete(child:Delay?) -> DelayBoundedDiscrete? {
    graft(child);
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
