/**
 * Lazy `rectify`.
 */
final class Rectify(y:Expression<Real>) <
    ScalarUnaryExpression<Expression<Real>,Real,Real,Real>(y) {
  override function doEvaluate(y:Real) -> Real {
    return rectify(y);
  }

  override function doEvaluateGrad(d:Real, x:Real, y:Real) -> Real {
    if x > 0.0 {
      return d;
    } else {
      return 0.0;
    }
  }
}

/**
 * Lazy `rectify`.
 */
function rectify(y:Expression<Real>) -> Rectify {
  return construct<Rectify>(y);
}
