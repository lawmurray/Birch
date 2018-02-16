/*
 * Logaritmic transformation of a random variable.
 */
class LogExpression {
  /**
   * Random variable.
   */
  x:Random<Real>;

  /**
   * Value conversion.
   */
  operator -> Real {
    return value();
  }

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
