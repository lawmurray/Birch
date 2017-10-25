/**
 * Expression that is a scaling of a log-Gaussian variable.
 */
class LogGaussianExpression {
  /**
   * Scale.
   */
  a:Real;
  
  /**
   * Base distribution.
   */
  x:LogGaussian;

  /**
   * Value conversion.
   */
  operator -> Real {
    return a*x.value();
  }
  
  /**
   * Initialize.
   */
  function initialize(a:Real, x:LogGaussian) {
    this.a <- a;
    this.x <- x;
  }
}

operator (a:Real*x:LogGaussian) -> LogGaussianExpression {
  y:LogGaussianExpression;
  y.initialize(a, x);
  return y;
}

operator (a:Real*x:LogGaussianExpression) -> LogGaussianExpression {
  y:LogGaussianExpression;
  y.initialize(a*x.a, x.x);
  return y;
}

operator (x:LogGaussian*a:Real) -> LogGaussianExpression {
  return a*x;
}

operator (x:LogGaussianExpression*a:Real) -> LogGaussianExpression {
  return a*x;
}

operator (x:LogGaussian/a:Real) -> LogGaussianExpression {
  return (1.0/a)*x;
}

operator (x:LogGaussianExpression/a:Real) -> LogGaussianExpression {
  return (1.0/a)*x;
}

function log(x:LogGaussianExpression) -> GaussianLogExpression {
  y:GaussianLogExpression;
  y.initialize(1.0, x.x, log(x.a));
  return y;
}
