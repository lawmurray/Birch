/*
 * Multivariate linear transformation.
 */
class TransformLinearMultivariate<Value>(A:Real[_,_], x:Value, c:Real[_]) {
  /**
   * Scale.
   */
  A:Real[_,_] <- A;
  
  /**
   * Delay node.
   */
  x:Value <- x;

  /**
   * Offset.
   */
  c:Real[_] <- c;
  
  function rows() -> Integer {
    return global.rows(A);
  }
  
  function leftMultiply(Y:Real[_,_]) {
    A <- Y*A;
    c <- Y*c;
  }

  function add(y:Real[_]) {
    c <- c + y;
  }

  function subtract(y:Real[_]) {
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
  
  function negateAndAdd(y:Real[_]) {
    A <- -A;
    c <- y - c;
  }
}

function TransformLinearMultivariate<Value>(A:Real[_,_], x:Value,
    c:Real[_]) -> TransformLinearMultivariate<Value> {
  m:TransformLinearMultivariate<Value>(A, x, c);
  return m;
}

function TransformLinearMultivariate<Value>(A:Real[_,_], x:Value) ->
    TransformLinearMultivariate<Value> {
  return TransformLinearMultivariate<Value>(A, x, vector(0.0, rows(A)));
}
