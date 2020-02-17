/*
 * Matrix linear transformation.
 */
class TransformLinearMatrix<Value>(A:Expression<Real[_,_]>, X:Value,
    C:Expression<Real[_,_]>) {
  /**
   * Scale.
   */
  A:Expression<Real[_,_]> <- A;
  
  /**
   * Delay node.
   */
  X:Value <- X;

  /**
   * Offset.
   */
  C:Expression<Real[_,_]> <- C;
    
  function rows() -> Integer {
    return global.rows(A);
  }
  
  function columns() -> Integer {
    return global.columns(C);
  }
    
  function leftMultiply(Y:Expression<Real[_,_]>) {
    A <- Y*A;
    C <- Y*C;
  }

  function add(Y:Expression<Real[_,_]>) {
    C <- C + Y;
  }

  function subtract(Y:Expression<Real[_,_]>) {
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
  
  function negateAndAdd(Y:Expression<Real[_,_]>) {
    A <- -A;
    C <- Y - C;
  }
}

function TransformLinearMatrix<Value>(A:Expression<Real[_,_]>, X:Value,
    C:Expression<Real[_,_]>) -> TransformLinearMatrix<Value> {
  m:TransformLinearMatrix<Value>(A, X, C);
  return m;
}

function TransformLinearMatrix<Value>(A:Expression<Real[_,_]>, X:Value) ->
    TransformLinearMatrix<Value> {
  return TransformLinearMatrix<Value>(A, X,
      Boxed(matrix(0.0, A.rows(), X.columns())));
}
