/*
 * Multivariate affine transformation of a random variable.
 */
class MultivariateAffineExpression {
  /**
   * Scale.
   */
  A:Real[_,_];
  
  /**
   * Random variable.
   */
  x:Random<Real[_]>;
  
  /**
   * Offset.
   */
  c:Real[_];

  /**
   * Value conversion.
   */
  operator -> Real[_] {
    return value();
  }

  /**
   * Value conversion.
   */
  function value() -> Real[_] {
    return A*x.value() + c;
  }
  
  /**
   * Initialize.
   */
  function initialize(A:Real[_,_], x:Random<Real[_]>, c:Real[_]) {
    this.A <- A;
    this.x <- x;
    this.c <- c;
  }
}

operator (x:Random<Real[_]> + c:Real[_]) -> MultivariateAffineExpression {
  y:MultivariateAffineExpression;
  y.initialize(identity(length(x)), x, c);
  return y;
}

operator (x:MultivariateAffineExpression + c:Real[_])
    -> MultivariateAffineExpression {
  y:MultivariateAffineExpression;
  y.initialize(x.A, x.x, x.c + c);
  return y;
}

operator (c:Real[_] + x:Random<Real[_]>) -> MultivariateAffineExpression {
  return x + c;
}

operator (c:Real[_] + x:MultivariateAffineExpression)
     -> MultivariateAffineExpression {
  return x + c;
}

operator (x:Random<Real[_]> - c:Real[_]) -> MultivariateAffineExpression {
  y:MultivariateAffineExpression;
  y.initialize(identity(length(x)), x, -c);
  return y;
}

operator (x:MultivariateAffineExpression - c:Real[_])
    -> MultivariateAffineExpression {
  y:MultivariateAffineExpression;
  y.initialize(x.A, x.x, x.c - c);
  return y;
}

operator (c:Real[_] - x:Random<Real[_]>) -> MultivariateAffineExpression {
  y:MultivariateAffineExpression;
  y.initialize(-identity(length(x)), x, c);
  return y;
}

operator (c:Real[_] - x:MultivariateAffineExpression)
    -> MultivariateAffineExpression {
  y:MultivariateAffineExpression;
  y.initialize(-x.A, x.x, c - x.c);
  return y;
}

operator (A:Real[_,_]*x:Random<Real[_]>) -> MultivariateAffineExpression {
  y:MultivariateAffineExpression;
  y.initialize(A, x, vector(0.0, rows(A)));
  return y;
}

operator (A:Real[_,_]*x:MultivariateAffineExpression)
    -> MultivariateAffineExpression {
  y:MultivariateAffineExpression;
  y.initialize(A*x.A, x.x, A*x.c);
  return y;
}
