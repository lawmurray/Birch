/**
 * Lazy `lchoose`.
 */
final class LogChoose<Left,Right,Value>(left:Left, right:Right) <
    ScalarBinaryExpression<Left,Right,Value>(left, right) {  
  override function doValue() {
    x <- lchoose(left!.value(), right!.value());
  }

  override function doPilot() {
    x <- lchoose(left!.pilot(), right!.pilot());
  }

  override function doMove(κ:Kernel) {
    x <- lchoose(left!.move(κ), right!.move(κ));
  }
  
  override function doGrad() {
    left!.grad(0.0);
    right!.grad(0.0);
  }
}

/**
 * Lazy `lchoose`.
 */
function lchoose(x:Expression<Integer>, y:Expression<Integer>) -> Expression<Real> {
  if x.isConstant() && y.isConstant() {
    return box(lchoose(x.value(), y.value()));
  } else {
    m:LogChoose<Expression<Integer>,Expression<Integer>,Real>(x, y);
    return m;
  }
}

/**
 * Lazy `lchoose`.
 */
function lchoose(x:Integer, y:Expression<Integer>) -> Expression<Real> {
  if y.isConstant() {
    return box(lchoose(x, y.value()));
  } else {
    return lchoose(box(x), y);
  }
}

/**
 * Lazy `lchoose`.
 */
function lchoose(x:Expression<Integer>, y:Integer) -> Expression<Real> {
  if x.isConstant() {
    return box(lchoose(x.value(), y));
  } else {
    return lchoose(x, box(y));
  }
}
