/**
 * Lazy `sin`.
 */
final class Sin(x:Expression<Real>) <
    ScalarUnaryExpression<Expression<Real>,Real>(x) {
  override function doValue() {
    x <- sin(single.value());
  }

  override function doPilot() {
    x <- sin(single.pilot());
  }

  override function doMove(κ:Kernel) {
    x <- sin(single.move(κ));
  }

  override function doGrad() {
    single.grad(d!*cos(single.get()));
  }
}

/**
 * Lazy `sin`.
 */
function sin(x:Expression<Real>) -> Expression<Real> {
  if x.isConstant() {
    return box(sin(x.value()));
  } else {
    m:Sin(x);
    return m;
  }
}
