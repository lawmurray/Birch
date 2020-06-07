/**
 * Lazy `lgamma`.
 */
final class LogGamma(x:Expression<Real>) <
    ScalarUnaryExpression<Expression<Real>,Real>(x) {
  override function doValue() {
    x <- lgamma(single.value());
  }

  override function doPilot() {
    x <- lgamma(single.pilot());
  }

  override function doMove(κ:Kernel) {
    x <- lgamma(single.move(κ));
  }

  override function doGrad() {
    single.grad(d!*digamma(single.get()));
  }
}

/**
 * Lazy `lgamma`.
 */
function lgamma(x:Expression<Real>) -> Expression<Real> {
  if x.isConstant() {
    return box(lgamma(x.value()));
  } else {
    m:LogGamma(x);
    return m;
  }
}
