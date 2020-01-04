/*
 * Multivariate linear transformation.
 */
class TransformLinearMultivariate<Value>(A:Expression<Real[_,_]>, x:Value,
    c:Expression<Real[_]>) {
  /**
   * Scale.
   */
  auto A <- A;
  
  /**
   * Delay node.
   */
  auto x <- x;

  /**
   * Offset.
   */
  auto c <- c;
  
  function rows() -> Integer {
    return global.rows(A);
  }
  
  function leftMultiply(Y:Expression<Real[_,_]>) {
    A <- Y*A;
    c <- Y*c;
  }

  function add(y:Expression<Real[_]>) {
    c <- c + y;
  }

  function subtract(y:Expression<Real[_]>) {
    c <- c - y;
  }

  function multiply(y:Real) {
    A <- A*y;
    c <- c*y;
  }
  
  function divide(y:Real) {
    A <- A/y;
    c <- c/y;
  }
  
  function negateAndAdd(y:Expression<Real[_]>) {
    A <- -A;
    c <- y - c;
  }
}

function TransformLinearMultivariate<Value>(A:Expression<Real[_,_]>,
    x:Value, c:Expression<Real[_]>) ->
    TransformLinearMultivariate<Value> {
  m:TransformLinearMultivariate<Value>(A, x, c);
  return m;
}

function TransformLinearMultivariate<Value>(A:Expression<Real[_,_]>,
    x:Value) -> TransformLinearMultivariate<Value> {
  return TransformLinearMultivariate<Value>(A, x, Boxed(vector(0.0, rows(A))));
}
