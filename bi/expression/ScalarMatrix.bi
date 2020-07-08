/**
 * Lazy `scalar`.
 */
final class ScalarMatrix(x:Expression<Real[_,_]>) <
    ScalarUnaryExpression<Expression<Real[_,_]>,Real>(x) {
  override function doValue() {
    x <- scalar(single!.value());
  }

  override function doPilot() {
    x <- scalar(single!.pilot());
  }

  override function doMove(κ:Kernel) {
    x <- scalar(single!.move(κ));
  }

  override function doGrad() {
    single!.grad([[d!]]);
  }
}

/**
 * Lazy `scalar`.
 */
function scalar(x:Expression<Real[_,_]>) -> Expression<Real> {
  if x.isConstant() {
    return box(scalar(x.value()));
  } else {
    return construct<ScalarMatrix>(x);
  }
}
