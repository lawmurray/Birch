/**
 * Binomial distribution.
 */
class Binomial(n:Expression<Integer>, ρ:Expression<Real>) < Random<Integer> {
  /**
   * Number of trials.
   */
  n:Expression<Integer> <- n;

  /**
   * Probability of a true result.
   */
  ρ:Expression<Real> <- ρ;

  function doParent() -> Delay? {
    if (ρ.isBeta()) {
      return ρ;
    } else {
      return nil;
    }
  }

  function doSimulate() -> Integer {
    if (ρ.isBeta()) {
      α:Real;
      β:Real;
      (α, β) <- ρ.getBeta();
      return simulate_beta_binomial(n.value(), α, β);
    } else {
      return simulate_binomial(n.value(), ρ.value());
    }
  }
  
  function doObserve(x:Integer) -> Real {
    if (ρ.isBeta()) {
      α:Real;
      β:Real;
      (α, β) <- ρ.getBeta();
      return observe_beta_binomial(x, n.value(), α, β);
    } else {
      return observe_binomial(x, n.value(), ρ.value());
    }
  }
}

/**
 * Create binomial distribution.
 */
function Binomial(n:Expression<Integer>, ρ:Expression<Real>) -> Binomial {
  m:Binomial(n, ρ);
  m.initialize();
  return m;
}

/**
 * Create binomial distribution.
 */
function Binomial(n:Expression<Integer>, ρ:Real) -> Binomial {
  return Binomial(n, Literal(ρ));
}

/**
 * Create binomial distribution.
 */
function Binomial(n:Integer, ρ:Expression<Real>) -> Binomial {
  return Binomial(Literal(n), ρ);
}

/**
 * Create binomial distribution.
 */
function Binomial(n:Integer, ρ:Real) -> Binomial {
  return Binomial(Literal(n), Literal(ρ));
}
