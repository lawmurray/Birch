/**
 * Expression that is an affine transformation of a multivariate Gaussian.
 *
 * `R` Number of rows in transformation.
 * `C` Number of columns in transformation.
 */
class MultivariateGaussianExpression(R:Integer, C:Integer) {
  /**
   * Scale.
   */
  A:Real[R,C];
  
  /**
   * Base distribution.
   */
  x:MultivariateGaussian(C);
  
  /**
   * Offset.
   */
  c:Real[R];

  /**
   * Value conversion.
   */
  operator -> Real[_] {
    return A*x.value() + c;
  }
  
  /**
   * Initialize.
   */
  function initialize(A:Real[_,_], x:MultivariateGaussian, c:Real[_]) {
    this.A <- A;
    this.x <- x;
    this.c <- c;
  }
}

operator (x:MultivariateGaussian + c:Real[_])
    -> MultivariateGaussianExpression {
  assert x.D == length(c);
  y:MultivariateGaussianExpression(x.D, x.D);
  y.initialize(I(x.D, x.D), x, c);
  return y;
}

operator (x:MultivariateGaussianExpression + c:Real[_])
    -> MultivariateGaussianExpression {
  y:MultivariateGaussianExpression(x.R, x.C);
  y.initialize(x.A, x.x, x.c + c);
  return y;
}

operator (c:Real[_] + x:MultivariateGaussian)
    -> MultivariateGaussianExpression {
  return x + c;
}

operator (c:Real[_] + x:MultivariateGaussianExpression)
    -> MultivariateGaussianExpression {
  return x + c;
}

operator (x:MultivariateGaussian - c:Real[_])
    -> MultivariateGaussianExpression {
  assert x.D == length(c);
  y:MultivariateGaussianExpression(x.D, x.D);
  y.initialize(I(x.D, x.D), x, -c);
  return y;
}

operator (x:MultivariateGaussianExpression - c:Real[_])
    -> MultivariateGaussianExpression {
  y:MultivariateGaussianExpression(x.R, x.C);
  y.initialize(x.A, x.x, x.c - c);
  return y;
}

operator (c:Real[_] - x:MultivariateGaussian)
    -> MultivariateGaussianExpression {
  assert x.D == length(c);
  y:MultivariateGaussianExpression(x.D, x.D);
  y.initialize(-I(x.D, x.D), x, c);
  return y;
}

operator (c:Real[_] - x:MultivariateGaussianExpression)
    -> MultivariateGaussianExpression {
  y:MultivariateGaussianExpression(x.R, x.C);
  y.initialize(-x.A, x.x, c - x.c);
  return y;
}

operator (A:Real[_,_]*x:MultivariateGaussian)
    -> MultivariateGaussianExpression {
  assert columns(A) == x.D;
  y:MultivariateGaussianExpression(rows(A), columns(A));
  y.initialize(A, x, vector(0.0, rows(A)));
  return y;
}

operator (A:Real[_,_]*x:MultivariateGaussianExpression)
    -> MultivariateGaussianExpression {
  y:MultivariateGaussianExpression(rows(A), x.C);
  y.initialize(A*x.A, x.x, A*x.c);
  return y;
}
