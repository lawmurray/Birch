/**
 * Lazy `rectify`.
 */
final class MatrixRectify(x:Expression<Real[_,_]>) <
    MatrixUnaryExpression<Expression<Real[_,_]>,Real[_,_]>(x) {
  override function doValue() {
    x <- transform(single.value(), \(y:Real) -> Real { return rectify(y); });
  }

  override function doGet() {
    x <- transform(single.get(), \(y:Real) -> Real { return rectify(y); });
  }

  override function doPilot() {
    x <- transform(single.pilot(), \(y:Real) -> Real { return rectify(y); });
  }

  override function doMove(κ:Kernel) {
    x <- transform(single.move(κ), \(y:Real) -> Real { return rectify(y); });
  }

  override function doGrad() {
    single.grad(transform(x!, d!, \(x:Real, d:Real) -> Real {
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
function rectify(x:Expression<Real[_,_]>) -> Expression<Real[_,_]> {
  if x.isConstant() {
    return box(transform(x.value(), \(y:Real) -> Real {
          return rectify(y);
        }));
  } else {
    m:MatrixRectify(x);
    return m;
  }
}
