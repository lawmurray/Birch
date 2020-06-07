/**
 * Lazy `scalar`.
 */
final class ScalarMultivariate(x:Expression<Real[_]>) <
    ScalarUnaryExpression<Expression<Real[_]>,Real>(x) {
  override function doValue() {
    x <- scalar(single.value());
  }

  override function doPilot() {
    x <- scalar(single.pilot());
  }

  override function doMove(κ:Kernel) {
    x <- scalar(single.move(κ));
  }

  override function doGrad() {
    single.grad([d!]);
  }
}

/**
 * Lazy `scalar`.
 */
function scalar(x:Expression<Real[_]>) -> Expression<Real> {
  if x.isConstant() {
    return box(scalar(x.value()));
  } else {
    m:ScalarMultivariate(x);
    return m;
  }
}
