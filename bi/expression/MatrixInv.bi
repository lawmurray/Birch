/**
 * Lazy `inv`.
 */
final class MatrixInv<Argument,ArgumentValue>(y:Argument) <
    MatrixUnaryExpression<Argument,ArgumentValue,Real[_,_],Real[_,_]>(y) {
  override function doRows() -> Integer {
    return y!.rows();
  }
  
  override function doColumns() -> Integer {
    return y!.columns();
  }

  override function doEvaluate(y:ArgumentValue) -> Real[_,_] {
    return inv(y);
  }

  override function doEvaluateGrad(d:Real[_,_], x:Real[_,_],
      y:ArgumentValue) -> Real[_,_] {
    return -transpose(x)*d*transpose(x);
  }
}

/**
 * Lazy `inv`.
 */
function inv(x:Expression<Real[_,_]>) ->
    MatrixInv<Expression<Real[_,_]>,Real[_,_]> {
  return construct<MatrixInv<Expression<Real[_,_]>,Real[_,_]>>(x);
}

/**
 * Lazy `inv`.
 */
function inv(x:Expression<LLT>) -> MatrixInv<Expression<LLT>,LLT> {
  return construct<MatrixInv<Expression<LLT>,LLT>>(x);
}
