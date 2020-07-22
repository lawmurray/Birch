/**
 * Lazy `exp`.
 */
final class Exp(y:Expression<Real>) <
    ScalarUnaryExpression<Expression<Real>,Real,Real,Real>(y) {
  override function doEvaluate(y:Real) -> Real {
    return exp(y);
  }

  override function doEvaluateGrad(d:Real, x:Real, y:Real) -> Real {
    return d*x;
  }
}

/**
 * Lazy `exp`.
 */
function exp(y:Expression<Real>) -> Exp {
  return construct<Exp>(y);
}
