/**
 * Binomial distribution.
 */
class Binomial<Type1,Type2>(n:Type1, ρ:Type2) < Random<Integer> {
  /**
   * Number of trials.
   */
  n:Type1 <- n;

  /**
   * Probability of a true result.
   */
  ρ:Type2 <- ρ;

  function doSimulate() -> Integer {
    return simulate_binomial(global.value(n), global.value(ρ));
  }
  
  function doObserve(x:Integer) -> Real {
    return observe_binomial(x, global.value(n), global.value(ρ));
  }
}

/**
 * Create binomial distribution.
 */
function Binomial(n:Integer, ρ:Real) -> Binomial<Integer,Real> {
  m:Binomial<Integer,Real>(n, ρ);
  m.initialize();
  return m;
}

/**
 * Create binomial distribution.
 */
function Binomial(n:Expression<Integer>, ρ:Real) ->
    Binomial<Expression<Integer>,Real> {
  m:Binomial<Expression<Integer>,Real>(n, ρ);
  m.initialize();
  return m;
}

/**
 * Create binomial distribution.
 */
function Binomial(n:Integer, ρ:Expression<Real>) ->
    Binomial<Integer,Expression<Real>> {
  m:Binomial<Integer,Expression<Real>>(n, ρ);
  m.initialize();
  return m;
}

/**
 * Create binomial distribution.
 */
function Binomial(n:Expression<Integer>, ρ:Expression<Real>) ->
    Binomial<Expression<Integer>,Expression<Real>> {
  m:Binomial<Expression<Integer>,Expression<Real>>(n, ρ);
  m.initialize();
  return m;
}
