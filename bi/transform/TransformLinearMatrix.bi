/*
 * Linear transformation of a matrix random variate, as represented by its
 * associated distribution.
 *
 * - Value: Distribution type.
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
    return C.rows();
  }
  
  function columns() -> Integer {
    return C.columns();
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

  /**
   * Is the transformation valid? This evaluates the scale and offset. It 
   * then returns true if the Distribution object remains uninstantiated, and
   * false otherwise (which would mean that either or both of the scale and
   * offset depend it).
   */
  function isValid() -> Boolean {
    A.value();
    C.value();
    return !X.hasValue();
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
