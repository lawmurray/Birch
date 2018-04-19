/**
 * Negative binomial distribution.
 */
class NegativeBinomial<Type1,Type2>(k:Type1, ρ:Type2) < Random<Integer> {
  /**
   * Number of successes before the experiment is stopped.
   */
  k:Type1 <- k;

  /**
   * Probability of success.
   */
  ρ:Type2 <- ρ;

  function update(k:Type1, ρ:Type2) {
    this.k <- k;
    this.ρ <- ρ;
  }

  function doSimulate() -> Integer {
    return simulate_negative_binomial(global.value(k), global.value(ρ));
  }
  
  function doObserve(x:Integer) -> Real {
    return observe_negative_binomial(x, global.value(k), global.value(ρ));
  }
}

/**
 * Create negative binomial distribution.
 */
function NegativeBinomial(k:Integer, ρ:Real) ->
    NegativeBinomial<Integer,Real> {
  m:NegativeBinomial<Integer,Real>(k, ρ);
  m.initialize();
  return m;
}

/**
 * Create negative binomial distribution.
 */
function NegativeBinomial(k:Expression<Integer>, ρ:Real) ->
    NegativeBinomial<Expression<Integer>,Real> {
  m:NegativeBinomial<Expression<Integer>,Real>(k, ρ);
  m.initialize();
  return m;
}

/**
 * Create negative binomial distribution.
 */
function NegativeBinomial(k:Integer, ρ:Expression<Real>) ->
    NegativeBinomial<Integer,Expression<Real>> {
  m:NegativeBinomial<Integer,Expression<Real>>(k, ρ);
  m.initialize();
  return m;
}

/**
 * Create negative binomial distribution.
 */
function NegativeBinomial(k:Expression<Integer>, ρ:Expression<Real>) ->
    NegativeBinomial<Expression<Integer>,Expression<Real>> {
  m:NegativeBinomial<Expression<Integer>,Expression<Real>>(k, ρ);
  m.initialize();
  return m;
}
