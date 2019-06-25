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

  function valueForward() -> Integer {
    assert !delay?;
    return simulate_binomial(n, ρ);
  }

  function observeForward(x:Integer) -> Real {
    assert !delay?;
    return logpdf_binomial(x, n, ρ);
  }

  function graft(force:Boolean) {
    if delay? {
      delay!.prune();
    } else {
      m:DelayBeta?;
      if (m <- ρ.graftBeta())? {
        delay <- DelayBetaBinomial(future, futureUpdate, n, m!);
      } else if force {
        delay <- DelayBinomial(future, futureUpdate, n, ρ);
      }
    }
  }
  
  function graftDiscrete() -> DelayDiscrete? {
    graft(true);
    return DelayDiscrete?(delay);
  }

  function graftBoundedDiscrete() -> DelayBoundedDiscrete? {
    graft(true);
    return DelayBoundedDiscrete?(delay);
  }

  function write(buffer:Buffer) {
    if delay? {
      delay!.write(buffer);
    } else {
      buffer.set("class", "Binomial");
      buffer.set("n", n.value());
      buffer.set("ρ", ρ.value());
    }
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
