/*
 * ed binomial random variate.
 */
final class Binomial(future:Integer?, futureUpdate:Boolean, n:Expression<Integer>, ρ:Expression<Real>) < BoundedDiscrete(future, futureUpdate, 0, n) {
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
      return simulate_binomial(n, ρ);
    }
  }
  
  function logpdf(x:Integer) -> Real {
    return logpdf_binomial(x, n, ρ);
  }

  function cdf(x:Integer) -> Real? {
    return cdf_binomial(x, n, ρ);
  }

  function quantile(p:Real) -> Integer? {
    return quantile_binomial(p, n, ρ);
  }

  function lower() -> Integer? {
    return 0;
  }
  
  function upper() -> Integer? {
    return n;
  }

  function graft() {
    if delay? {
      delay!.prune();
    } else {
      m:Beta?;
      if (m <- ρ.graftBeta())? {
        delay <- BetaBinomial(future, futureUpdate, n, m!);
      } else {
        delay <- Binomial(future, futureUpdate, n, ρ);
      }
    }
  }
  
  function graftDiscrete() -> Discrete? {
    return graftBoundedDiscrete();
  }

  function graftBoundedDiscrete() -> BoundedDiscrete? {
    graft();
    return BoundedDiscrete?(delay);
  }

  function write(buffer:Buffer) {
    prune();
    buffer.set("class", "Binomial");
    buffer.set("n", n);
    buffer.set("ρ", ρ);
  }
}

function Binomial(future:Integer?, futureUpdate:Boolean, n:Integer, ρ:Real) ->
    Binomial {
  m:Binomial(future, futureUpdate, n, ρ);
  return m;
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
