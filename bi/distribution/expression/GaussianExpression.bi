/**
 * Expression that is an affine transformation of a Gaussian variable.
 */
class GaussianExpression {  
  /**
   * Scale.
   */
  a:Real;
  
  /**
   * Base distribution.
   */
  x:Gaussian;
  
  /**
   * Offset.
   */
  c:Real;

  /**
   * Value conversion.
   */
  operator -> Real {
    return a*x.value() + c;
  }
  
  /**
   * Initialize.
   */
  function initialize(a:Real, x:Gaussian, c:Real) {
    this.a <- a;
    this.x <- x;
    this.c <- c;
  }
}

operator (+x:Gaussian) -> Gaussian {
  return x;
}

operator (+x:GaussianExpression) -> GaussianExpression {
  return x;
}

operator (-x:Gaussian) -> GaussianExpression {
  y:GaussianExpression;
  y.initialize(-1.0, x, 0.0);
  return y;
}

operator (-x:GaussianExpression) -> GaussianExpression {
  y:GaussianExpression;
  y.initialize(-x.a, x.x, -x.c);
  return y;
}

operator (x:Gaussian + c:Real) -> GaussianExpression {
  y:GaussianExpression;
  y.initialize(1.0, x, c);
  return y;
}

operator (x:GaussianExpression + c:Real) -> GaussianExpression {
  y:GaussianExpression;
  y.initialize(x.a, x.x, x.c + c);
  return y;
}

operator (c:Real + x:Gaussian) -> GaussianExpression {
  return x + c;
}

operator (c:Real + x:GaussianExpression) -> GaussianExpression {
  return x + c;
}

operator (x:Gaussian - c:Real) -> GaussianExpression {
  y:GaussianExpression;
  y.initialize(1.0, x, -c);
  return y;
}

operator (x:GaussianExpression - c:Real) -> GaussianExpression {
  y:GaussianExpression;
  y.initialize(x.a, x.x, x.c - c);
  return y;
}

operator (c:Real - x:Gaussian) -> GaussianExpression {
  y:GaussianExpression;
  y.initialize(-1.0, x, c);
  return y;
}

operator (c:Real - x:GaussianExpression) -> GaussianExpression {
  y:GaussianExpression;
  y.initialize(-x.a, x.x, c - x.c);
  return y;
}

operator (a:Real*x:Gaussian) -> GaussianExpression {
  y:GaussianExpression;
  y.initialize(a, x, 0.0);
  return y;
}

operator (a:Real*x:GaussianExpression) -> GaussianExpression {
  y:GaussianExpression;
  y.initialize(a*x.a, x.x, a*x.c);
  return y;
}

operator (x:Gaussian*a:Real) -> GaussianExpression {
  return a*x;
}

operator (x:GaussianExpression*a:Real) -> GaussianExpression {
  return a*x;
}

operator (x:Gaussian/a:Real) -> GaussianExpression {
  return (1.0/a)*x;
}

operator (x:GaussianExpression/a:Real) -> GaussianExpression {
  return (1.0/a)*x;
}
