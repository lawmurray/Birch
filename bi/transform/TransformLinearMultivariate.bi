/*
 * Linear transformation of a multivariate random variate, as represented by
 * its associated distribution.
 *
 * - Value: Distribution type.
 */
class TransformLinearMultivariate<Value>(A:Expression<Real[_,_]>, x:Value,
    c:Expression<Real[_]>) {
  /**
   * Scale.
   */
  A:Expression<Real[_,_]> <- A;
  
  /**
   * Delay node.
   */
  x:Value <- x;

  /**
   * Offset.
   */
  c:Expression<Real[_]> <- c;
  
  function rows() -> Integer {
    return global.rows(A);
  }
  
  function leftMultiply(Y:Expression<Real[_,_]>) {
    A <- Y*A;
    c <- Y*c;
  }

  function leftMultiply(y:Expression<Real>) {
    A <- y*A;
    c <- y*c;
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

  function negate() {
    A <- -A;
    c <- -c;
  }
  
  function negateAndAdd(y:Expression<Real[_]>) {
    A <- -A;
    c <- y - c;
  }
}

function TransformLinearMultivariate<Value>(A:Expression<Real[_,_]>,
    x:Value, c:Expression<Real[_]>) ->
    TransformLinearMultivariate<Value> {
  return construct<TransformLinearMultivariate<Value>>(A, x, c);
}

function TransformLinearMultivariate<Value>(A:Expression<Real[_,_]>,
    x:Value) -> TransformLinearMultivariate<Value> {
  return TransformLinearMultivariate<Value>(A, x, box(vector(0.0, A.rows())));
}
