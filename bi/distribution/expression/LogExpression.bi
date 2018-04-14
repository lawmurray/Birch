/*
 * Logaritmic transformation of a random variable.
 */
class LogExpression < Expression<Real> {
  /**
   * Random variable.
   */
  x:Random<Real>;

  /**
   * Value conversion.
   */
  function value() -> Real {
    return log(x.value());
  }
  
  /**
   * Initialize.
   */
  function initialize(x:Random<Real>) {
    this.x <- x;
  }
}

function log(x:Random<Real>) -> LogExpression {
  y:LogExpression;
  y.initialize(x);
  return y;
}
