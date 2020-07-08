/**
 * Lazy `sinh`.
 */
final class Sinh(x:Expression<Real>) <
    ScalarUnaryExpression<Expression<Real>,Real>(x) {
  override function doValue() {
    x <- sinh(single!.value());
  }

  override function doPilot() {
    x <- sinh(single!.pilot());
  }

  override function doMove(κ:Kernel) {
    x <- sinh(single!.move(κ));
  }

  override function doGrad() {
    single!.grad(d!*cosh(single!.get()));
  }
}

/**
 * Lazy `sinh`.
 */
function sinh(x:Expression<Real>) -> Expression<Real> {
  if x.isConstant() {
    return box(sinh(x.value()));
  } else {
    return construct<Sinh>(x);
  }
}
