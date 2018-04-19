/**
 * Multinomial distribution.
 */
class Multinomial<Type1,Type2>(n:Type1, ρ:Type2) < Random<Integer[_]> {
  /**
   * Number of trials.
   */
  n:Type1;

  /**
   * Category probabilities.
   */
  ρ:Type2;

  function update(n:Type1, ρ:Type2) {
    this.n <- n;
    this.ρ <- ρ;
  }

  function doSimulate() -> Integer[_] {
    return simulate_multinomial(global.value(n), global.value(ρ));
  }
  
  function doObserve(x:Integer[_]) -> Real {
    return observe_multinomial(x, global.value(n), global.value(ρ));
  }
}

/**
 * Create multinomial distribution.
 */
function Multinomial(n:Integer, ρ:Real[_]) -> Multinomial<Integer,Real[_]> {
  m:Multinomial<Integer,Real[_]>(n, ρ);
  m.initialize();
  return m;
}

/**
 * Create multinomial distribution.
 */
function Multinomial(n:Expression<Integer>, ρ:Real[_]) ->
    Multinomial<Expression<Integer>,Real[_]> {
  m:Multinomial<Expression<Integer>,Real[_]>(n, ρ);
  m.initialize();
  return m;
}

/**
 * Create multinomial distribution.
 */
function Multinomial(n:Integer, ρ:Expression<Real[_]>) ->
    Multinomial<Integer,Expression<Real[_]>> {
  m:Multinomial<Integer,Expression<Real[_]>>(n, ρ);
  m.initialize();
  return m;
}

/**
 * Create multinomial distribution.
 */
function Multinomial(n:Expression<Integer>, ρ:Expression<Real[_]>) -> 
    Multinomial<Expression<Integer>,Expression<Real[_]>> {
  m:Multinomial<Expression<Integer>,Expression<Real[_]>>(n, ρ);
  m.initialize();
  return m;
}
