/**
 * Lazy `cosh`.
 */
final class Cosh(x:Expression<Real>) <
    ScalarUnaryExpression<Expression<Real>,Real>(x) {
  override function doValue() {
    x <- cosh(single!.value());
  }

  override function doPilot() {
    x <- cosh(single!.pilot());
  }

  override function doMove(κ:Kernel) {
    x <- cosh(single!.move(κ));
  }

  override function doGrad() {
    single!.grad(-d!*sinh(single!.get()));
  }
}

/**
 * Lazy `cosh`.
 */
function cosh(x:Expression<Real>) -> Expression<Real> {
  if x.isConstant() {
    return box(cosh(x.value()));
  } else {
    m:Cosh(x);
    return m;
  }
}
