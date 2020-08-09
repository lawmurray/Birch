/**
 * Lazy `llt`.
 */
final class MatrixLLT(y:Expression<Real[_,_]>) <
    MatrixUnaryExpression<Expression<Real[_,_]>,Real[_,_],Real[_,_],LLT>(y) {
  override function doRows() -> Integer {
    return y!.rows();
  }
  
  override function doColumns() -> Integer {
    return y!.columns();
  }

  override function doEvaluate(y:Real[_,_]) -> LLT {
    return llt(y);
  }

  override function doEvaluateGrad(d:Real[_,_], x:LLT, y:Real[_,_]) ->
      Real[_,_] {
    return d;  // just a factorization, so pass along
  }
}

/**
 * Lazy `inv`.
 */
function llt(y:Expression<Real[_,_]>) -> MatrixLLT {
  return construct<MatrixLLT>(y);
}
