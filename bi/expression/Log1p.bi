/**
 * Lazy `log1p`.
 */
final class Log1p(x:Expression<Real>) <
    ScalarUnaryExpression<Expression<Real>,Real>(x) {
  override function doValue() {
    x <- log1p(single!.value());
  }

  override function doPilot() {
    x <- log1p(single!.pilot());
  }

  override function doMove(κ:Kernel) {
    x <- log1p(single!.move(κ));
  }

  override function doGrad() {
    single!.grad(d!/(1.0 + single!.get()));
  }
}

/**
 * Lazy `log1p`.
 */
function log1p(x:Expression<Real>) -> Expression<Real> {
  if x.isConstant() {
    return box(log1p(x.value()));
  } else {
    return construct<Log1p>(x);
  }
}
