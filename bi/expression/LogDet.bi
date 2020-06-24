/**
 * Lazy `ldet`.
 */
final class LogDet<Single,Value>(x:Single) <
    ScalarUnaryExpression<Single,Value>(x) {
  override function doValue() {
    x <- ldet(single!.value());
  }

  override function doPilot() {
    x <- ldet(single!.pilot());
  }

  override function doMove(κ:Kernel) {
    x <- ldet(single!.move(κ));
  }

  override function doGrad() {
    single!.grad(matrix(inv(transpose(single!.get()))));
  }
}

/**
 * Lazy `ldet`.
 */
function ldet(x:Expression<LLT>) -> Expression<Real> {
  if x.isConstant() {
    return box(ldet(x.value()));
  } else {
    m:LogDet<Expression<LLT>,Real>(x);
    return m;
  }
}

/**
 * Lazy `ldet`.
 */
function ldet(x:Expression<Real[_,_]>) -> Expression<Real> {
  if x.isConstant() {
    return box(ldet(x.value()));
  } else {
    m:LogDet<Expression<Real[_,_]>,Real>(x);
    return m;
  }
}
