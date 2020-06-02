/**
 * Binomial distribution.
 */
class Binomial(n:Expression<Integer>, ρ:Expression<Real>) < BoundedDiscrete {
  /**
   * Number of trials.
   */
  n:Expression<Integer> <- n;

  /**
   * Success probability.
   */
  ρ:Expression<Real> <- ρ;

  function simulate() -> Integer {
    if value? {
      return value!;
    } else {
      return simulate_binomial(n.value(), ρ.value());
    }
  }
  
  function logpdf(x:Integer) -> Real {
    return logpdf_binomial(x, n.value(), ρ.value());
  }

  function cdf(x:Integer) -> Real? {
    return cdf_binomial(x, n.value(), ρ.value());
  }

  function quantile(P:Real) -> Integer? {
    return quantile_binomial(P, n.value(), ρ.value());
  }

  function lower() -> Integer? {
    return 0;
  }
  
  function upper() -> Integer? {
    return n.value();
  }

  function graft() -> Distribution<Integer> {
    prune();
    m:Beta?;
    r:Distribution<Integer> <- this;
    
    /* match a template */
    if (m <- ρ.graftBeta())? {
      r <- BetaBinomial(n, m!);
    }
    
    return r;
  }
  
  function graftDiscrete() -> Discrete? {
    return Discrete?(graft());
  }

  function graftBoundedDiscrete() -> BoundedDiscrete? {
    return BoundedDiscrete?(graft());
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "Binomial");
    buffer.set("n", n);
    buffer.set("ρ", ρ);
  }
}

/**
 * Create binomial distribution.
 */
function Binomial(n:Expression<Integer>, ρ:Expression<Real>) -> Binomial {
  m:Binomial(n, ρ);
  m.link();
  return m;
}

/**
 * Create binomial distribution.
 */
function Binomial(n:Expression<Integer>, ρ:Real) -> Binomial {
  return Binomial(n, box(ρ));
}

/**
 * Create binomial distribution.
 */
function Binomial(n:Integer, ρ:Expression<Real>) -> Binomial {
  return Binomial(box(n), ρ);
}

/**
 * Create binomial distribution.
 */
function Binomial(n:Integer, ρ:Real) -> Binomial {
  return Binomial(box(n), box(ρ));
}
