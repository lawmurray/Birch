/**
 * Lazy `ldet`.
 */
final class LogDet<Argument,ArgumentValue>(y:Argument) <
    ScalarUnaryExpression<Argument,ArgumentValue,Real[_,_],Real>(y) {
  override function doEvaluate(y:ArgumentValue) -> Real {
    return ldet(y);
  }

  override function doEvaluateGrad(d:Real, x:Real, y:ArgumentValue) ->
      Real[_,_] {
    return d*matrix(inv(transpose(y)));
  }
}

/**
 * Lazy `ldet`.
 */
function ldet(y:Expression<LLT>) -> LogDet<Expression<LLT>,LLT> {
  return construct<LogDet<Expression<LLT>,LLT>>(y);
}

/**
 * Lazy `ldet`.
 */
function ldet(y:Expression<Real[_,_]>) ->
    LogDet<Expression<Real[_,_]>,Real[_,_]> {
  return construct<LogDet<Expression<Real[_,_]>,Real[_,_]>>(y);
}
