/**
 * Affine transformation of the logarithm of a random variable.
 */
class AffineLogExpression {
  /**
   * Scale.
   */
  a:Real;
  
  /**
   * Logarithmic transform of the random variable.
   */
  x:LogExpression;
  
  /**
   * Offset.
   */
  c:Real;

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
    return a*log(x.value()) + c;
  }
  
  /**
   * Initialize.
   */
  function initialize(a:Real, x:LogExpression, c:Real) {
    this.a <- a;
    this.x <- x;
    this.c <- c;
  }
}

operator (+x:AffineLogExpression) -> AffineLogExpression {
  return x;
}

operator (+x:LogExpression) -> LogExpression {
  return x;
}

operator (-x:AffineLogExpression) -> AffineLogExpression {
  y:AffineLogExpression;
  y.initialize(-x.a, x.x, -x.c);
  return y;
}

operator (-x:LogExpression) -> AffineLogExpression {
  y:AffineLogExpression;
  y.initialize(-1.0, x, 0.0);
  return y;
}

operator (x:AffineLogExpression + c:Real) -> AffineLogExpression {
  y:AffineLogExpression;
  y.initialize(x.a, x.x, x.c + c);
  return y;
}

operator (x:LogExpression + c:Real) -> AffineLogExpression {
  y:AffineLogExpression;
  y.initialize(0.0, x, c);
  return y;
}

operator (c:Real + x:AffineLogExpression) -> AffineLogExpression {
  return x + c;
}

operator (c:Real + x:LogExpression) -> AffineLogExpression {
  return x + c;
}

operator (x:AffineLogExpression - c:Real) -> AffineLogExpression {
  y:AffineLogExpression;
  y.initialize(x.a, x.x, x.c - c);
  return y;
}

operator (x:LogExpression - c:Real) -> AffineLogExpression {
  y:AffineLogExpression;
  y.initialize(1.0, x, -c);
  return y;
}

operator (c:Real - x:AffineLogExpression) -> AffineLogExpression {
  y:AffineLogExpression;
  y.initialize(-x.a, x.x, c - x.c);
  return y;
}

operator (c:Real - x:LogExpression) -> AffineLogExpression {
  y:AffineLogExpression;
  y.initialize(-1.0, x, c);
  return y;
}

operator (a:Real*x:AffineLogExpression) -> AffineLogExpression {
  y:AffineLogExpression;
  y.initialize(a*x.a, x.x, a*x.c);
  return y;
}

operator (a:Real*x:LogExpression) -> AffineLogExpression {
  y:AffineLogExpression;
  y.initialize(a, x, 0.0);
  return y;
}

operator (x:AffineLogExpression*a:Real) -> AffineLogExpression {
  return a*x;
}

operator (x:LogExpression*a:Real) -> AffineLogExpression {
  return a*x;
}

operator (x:AffineLogExpression/a:Real) -> AffineLogExpression {
  return (1.0/a)*x;
}

operator (x:LogExpression/a:Real) -> AffineLogExpression {
  return (1.0/a)*x;
}
