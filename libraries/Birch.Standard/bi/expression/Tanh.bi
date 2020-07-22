/**
 * Lazy `tanh`.
 */
final class Tanh(y:Expression<Real>) <
    ScalarUnaryExpression<Expression<Real>,Real,Real,Real>(y) {
  override function doEvaluate(y:Real) -> Real {
    return tanh(y);
  }

  override function doEvaluateGrad(d:Real, x:Real, y:Real) -> Real {
    return d*(1.0 + pow(tanh(y), 2.0));
  }
}

/**
 * Lazy `tanh`.
 */
function tanh(x:Expression<Real>) -> Tanh {
  return construct<Tanh>(x);
}
