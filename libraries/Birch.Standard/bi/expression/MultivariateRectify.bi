/**
 * Lazy `rectify`.
 */
final class MultivariateRectify(y:Expression<Real[_]>) <
    MultivariateUnaryExpression<Expression<Real[_]>,Real[_],Real[_],
    Real[_]>(y) {
  override function doRows() -> Integer {
    return y!.rows();
  }
    
  override function doEvaluate(y:Real[_]) -> Real[_] {
    return transform(y, \(y:Real) -> Real { return rectify(y); });
  }

  override function doEvaluateGrad(d:Real[_], x:Real[_], y:Real[_]) ->
      Real[_] {
    return transform(d, x, \(d:Real, x:Real) -> Real {
        if x > 0.0 {
          return d;
        } else {
          return 0.0;
        }
      });
  }
}

/**
 * Lazy `rectify`.
 */
function rectify(x:Expression<Real[_]>) -> MultivariateRectify {
  return construct<MultivariateRectify>(x);
}
