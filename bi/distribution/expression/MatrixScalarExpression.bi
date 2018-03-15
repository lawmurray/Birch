/*
 * Matrix scaling of a scalar random variable.
 */
class MatrixScalarExpression {
  /**
   * Scale.
   */
  A:Real[_,_];
  
  /**
   * Random variable.
   */
  x:Random<Real>;

  /**
   * Value conversion.
   */
  operator -> Real[_,_] {
    return value();
  }

  /**
   * Value conversion.
   */
  function value() -> Real[_,_] {
    return A*x.value();
  }
  
  /**
   * Initialize.
   */
  function initialize(A:Real[_,_], x:Random<Real>) {
    this.A <- A;
    this.x <- x;
  }
}

operator (A:Real[_,_]*x:Random<Real>) -> MatrixScalarExpression {
  y:MatrixScalarExpression;
  y.initialize(A, x);
  return y;
}

operator (x:Random<Real>*A:Real[_,_]) -> MatrixScalarExpression {
  return A*x;
}

operator (A:Real[_,_]*x:MatrixScalarExpression) -> MatrixScalarExpression {
  y:MatrixScalarExpression;
  y.initialize(A*x.A, x.x);
  return y;
}

operator (x:MatrixScalarExpression*A:Real[_,_]) -> MatrixScalarExpression {
  return A*x;
}
