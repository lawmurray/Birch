/**
 * Lazy `matrix`.
 */
final class MatrixMatrix<Argument,ArgumentValue>(y:Argument) <
    MatrixUnaryExpression<Argument,ArgumentValue,Real[_,_],Real[_,_]>(y) {
  override function doRows() -> Integer {
    return y!.rows();
  }
  
  override function doColumns() -> Integer {
    return y!.columns();
  }

  override function doEvaluate(y:ArgumentValue) -> Real[_,_] {
    return matrix(y);
  }

  override function doEvaluateGrad(d:Real[_,_], x:Real[_,_],
      y:ArgumentValue) -> Real[_,_] {
    return d;
  }
}

/**
 * Lazy `matrix`.
 */
function matrix(y:Expression<LLT>) -> Expression<Real[_,_]> {
  return construct<MatrixMatrix<Expression<LLT>,LLT>>(y);
}

/**
 * Lazy `matrix`.
 */
function matrix(y:Expression<Real[_,_]>) -> Expression<Real[_,_]> {
  if !y.isRandom() {
    /* just an identity function */
    return y;
  } else {  
    /* Random objects should be wrapped to allow the accumulation of
     * gradients by element if necessary; see note in split() also */
    return construct<MatrixMatrix<Expression<Real[_,_]>,Real[_,_]>>(y);
  }
}
