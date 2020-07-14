/**
 * Lazy `tan`.
 */
final class Tan(y:Expression<Real>) <
    ScalarUnaryExpression<Expression<Real>,Real,Real,Real>(y) {
  override function doEvaluate(y:Real) -> Real {
    return tan(y);
  }

  override function doEvaluateGrad(d:Real, x:Real, y:Real) -> Real {
    return d*(1.0 + pow(tan(y), 2.0));
  }
}

/**
 * Lazy `tan`.
 */
function tan(y:Expression<Real>) -> Tan {
  return construct<Tan>(y);
}
