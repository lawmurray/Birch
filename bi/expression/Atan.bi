/**
 * Lazy `atan`.
 */
final class Atan(x:Expression<Real>) <
    ScalarUnaryExpression<Expression<Real>,Real>(x) {
  override function doValue() {
    x <- atan(single!.value());
  }

  override function doPilot() {
    x <- atan(single!.pilot());
  }

  override function doMove(κ:Kernel) {
    x <- atan(single!.move(κ));
  }

  override function doGrad() {
    auto x <- single!.get();
    single!.grad(d!/(1.0 + x*x));
  }
}

/**
 * Lazy `atan`.
 */
function atan(x:Expression<Real>) -> Expression<Real> {
  if x.isConstant() {
    return box(atan(x.value()));
  } else {
    m:Atan(x);
    return m;
  }
}
