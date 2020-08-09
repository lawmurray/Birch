/**
 * Lazy `lgamma`.
 */
final class LogGamma(x:Expression<Real>) <
    ScalarUnaryExpression<Expression<Real>,Real,Real,Real>(x) {
  override function doEvaluate(y:Real) -> Real {
    return lgamma(y);
  }

  override function doEvaluateGrad(d:Real, x:Real, y:Real) -> Real {
    return d*digamma(y);
  }
}

/**
 * Lazy `lgamma`.
 */
function lgamma(x:Expression<Real>) -> LogGamma {
  return construct<LogGamma>(x);
}
