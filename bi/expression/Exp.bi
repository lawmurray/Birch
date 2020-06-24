/**
 * Lazy `exp`.
 */
final class Exp(x:Expression<Real>) <
    ScalarUnaryExpression<Expression<Real>,Real>(x) {
  override function doValue() {
    x <- exp(single!.value());
  }

  override function doPilot() {
    x <- exp(single!.pilot());
  }

  override function doMove(κ:Kernel) {
    x <- exp(single!.move(κ));
  }

  override function doGrad() {
    single!.grad(d!*x!);
  }
}

/**
 * Lazy `exp`.
 */
function exp(x:Expression<Real>) -> Expression<Real> {
  if x.isConstant() {
    return box(exp(x.value()));
  } else {
    m:Exp(x);
    return m;
  }
}
