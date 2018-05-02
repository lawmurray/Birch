/**
 * Binomial distribution.
 */
class Binomial(n:Expression<Integer>, ρ:Expression<Real>) < Random<Integer> {
  /**
   * Number of trials.
   */
  n:Expression<Integer> <- n;

  /**
   * Success probability.
   */
  ρ:Expression<Real> <- ρ;

  function doGraft() -> DelayValue<Integer>? {
    m:DelayBeta?;
    if (m <- ρ.graftBeta())? {
      return DelayBetaBinomial(this, n, m!);
    } else {
      return DelayBinomial(this, n, ρ);
    }
  }
  
  function doGraftBinomial() -> DelayBinomial? {
    return DelayBinomial(this, n, ρ);
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
