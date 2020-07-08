/**
 * Lazy `cos`.
 */
final class Cos(x:Expression<Real>) <
    ScalarUnaryExpression<Expression<Real>,Real>(x) {
  override function doValue() {
    x <- cos(single!.value());
  }

  override function doPilot() {
    x <- cos(single!.pilot());
  }

  override function doMove(κ:Kernel) {
    x <- cos(single!.move(κ));
  }

  override function doGrad() {
    single!.grad(-d!*sin(single!.get()));
  }
}

/**
 * Lazy `cos`.
 */
function cos(x:Expression<Real>) -> Expression<Real> {
  if x.isConstant() {
    return box(cos(x.value()));
  } else {
    return construct<Cos>(x);
  }
}
