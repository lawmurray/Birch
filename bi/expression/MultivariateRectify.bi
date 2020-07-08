/**
 * Lazy `rectify`.
 */
final class MultivariateRectify(x:Expression<Real[_]>) <
    MultivariateUnaryExpression<Expression<Real[_]>,Real[_]>(x) {
  override function doRows() -> Integer {
    return single!.rows();
  }
    
  override function doValue() {
    x <- transform(single!.value(), \(y:Real) -> Real { return rectify(y); });
  }

  override function doPilot() {
    x <- transform(single!.pilot(), \(y:Real) -> Real { return rectify(y); });
  }

  override function doMove(κ:Kernel) {
    x <- transform(single!.move(κ), \(y:Real) -> Real { return rectify(y); });
  }

  override function doGrad() {
    single!.grad(transform(x!, d!, \(x:Real, d:Real) -> Real {
          if x > 0.0 {
            return d;
          } else {
            return 0.0;
          }
        }));
  }
}

/**
 * Lazy `rectify`.
 */
function rectify(x:Expression<Real[_]>) -> Expression<Real[_]> {
  if x.isConstant() {
    return box(transform(x.value(), \(y:Real) -> Real {
          return rectify(y);
        }));
  } else {
    return construct<MultivariateRectify>(x);
  }
}
