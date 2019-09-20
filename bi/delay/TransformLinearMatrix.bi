/*
 * Matrix linear transformation.
 */
class TransformLinearMatrix<Value>(A:Real[_,_], X:Value, C:Real[_,_]) {
  /**
   * Scale.
   */
  A:Real[_,_] <- A;
  
  /**
   * Delay node.
   */
  X:Value <- X;

  /**
   * Offset.
   */
  C:Real[_,_] <- C;
    
  function leftMultiply(Y:Real[_,_]) {
    A <- Y*A;
    C <- Y*C;
  }

  function add(Y:Real[_,_]) {
    C <- C + Y;
  }

  function subtract(Y:Real[_,_]) {
    C <- C - Y;
  }

  function multiply(y:Real) {
    A <- A*y;
    C <- C*y;
  }
  
  function divide(y:Real) {
    A <- A/y;
    C <- C/y;
  }
  
  function negateAndAdd(Y:Real[_,_]) {
    A <- -A;
    C <- Y - C;
  }
}

function TransformLinearMatrix<Value>(A:Real[_,_], X:Value,
    C:Real[_,_]) -> TransformLinearMatrix<Value> {
  m:TransformLinearMatrix<Value>(A, X, C);
  return m;
}

function TransformLinearMatrix<Value>(A:Real[_,_], X:Value) ->
    TransformLinearMatrix<Value> {
  return TransformLinearMatrix<Value>(A, X, matrix(0.0, rows(A), X.columns()));
}
