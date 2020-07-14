/**
 * Lazy `trace`.
 */
final class Trace<Argument,ArgumentValue>(y:Argument) <
    ScalarUnaryExpression<Argument,ArgumentValue,Real[_,_],Real>(y) {
  override function doEvaluate(y:ArgumentValue) -> Real {
    return trace(y);
  }

  override function doEvaluateGrad(d:Real, x:Real, y:ArgumentValue) ->
      Real[_,_] {
    return diagonal(d, global.rows(y));
  }
}

/**
 * Lazy `trace`.
 */
function trace(y:Expression<LLT>) -> Trace<Expression<LLT>,LLT> {
  return construct<Trace<Expression<LLT>,LLT>>(y);
}

/**
 * Lazy `trace`.
 */
function trace(y:Expression<Real[_,_]>) ->
    Trace<Expression<Real[_,_]>,Real[_,_]> {
  return construct<Trace<Expression<Real[_,_]>,Real[_,_]>>(y);
}
