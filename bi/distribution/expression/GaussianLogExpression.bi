/**
 * Expression that is an affine transformation of the logarithm of a
 * log-Gaussian variable.
 */
class GaussianLogExpression {
  /**
   * Scale.
   */
  a:Real;
  
  /**
   * Base distribution.
   */
  x:LogGaussian;
  
  /**
   * Offset.
   */
  c:Real;

  /**
   * Value conversion.
   */
  operator -> Real {
    return a*log(x.value()) + c;
  }
  
  /**
   * Initialize.
   */
  function initialize(a:Real, x:LogGaussian, c:Real) {
    this.a <- a;
    this.x <- x;
    this.c <- c;
  }
}

operator (+x:GaussianLogExpression) -> GaussianLogExpression {
  return x;
}

operator (-x:GaussianLogExpression) -> GaussianLogExpression {
  y:GaussianLogExpression;
  y.initialize(-x.a, x.x, -x.c);
  return y;
}

operator (x:GaussianLogExpression + c:Real) -> GaussianLogExpression {
  y:GaussianLogExpression;
  y.initialize(x.a, x.x, x.c + c);
  return y;
}

operator (c:Real + x:GaussianLogExpression) -> GaussianLogExpression {
  return x + c;
}

operator (x:GaussianLogExpression - c:Real) -> GaussianLogExpression {
  y:GaussianLogExpression;
  y.initialize(x.a, x.x, x.c - c);
  return y;
}

operator (c:Real - x:GaussianLogExpression) -> GaussianLogExpression {
  y:GaussianLogExpression;
  y.initialize(-x.a, x.x, c - x.c);
  return y;
}

operator (a:Real*x:GaussianLogExpression) -> GaussianLogExpression {
  y:GaussianLogExpression;
  y.initialize(a*x.a, x.x, a*x.c);
  return y;
}

operator (x:GaussianLogExpression*a:Real) -> GaussianLogExpression {
  return a*x;
}

operator (x:GaussianLogExpression/a:Real) -> GaussianLogExpression {
  return (1.0/a)*x;
}
