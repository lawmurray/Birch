/**
 * Lazy `sqrt`.
 */
final class Sqrt(x:Expression<Real>) <
    ScalarUnaryExpression<Expression<Real>,Real>(x) {
  override function doValue() {
    x <- sqrt(single!.value());
  }

  override function doPilot() {
    x <- sqrt(single!.pilot());
  }

  override function doMove(κ:Kernel) {
    x <- sqrt(single!.move(κ));
  }

  override function doGrad() {
    single!.grad(d!*0.5/sqrt(single!.get()));
  }
}

/**
 * Lazy `sqrt`.
 */
function sqrt(x:Expression<Real>) -> Expression<Real> {
  if x.isConstant() {
    return box(sqrt(x.value()));
  } else {
    return construct<Sqrt>(x);
  }
}
