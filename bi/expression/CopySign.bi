/**
 * Lazy `copysign`.
 */
final class CopySign<Left,Right,Value>(left:Left, right:Right) <
    ScalarBinaryExpression<Left,Right,Value>(left, right) {  
  override function doValue() {
    x <- copysign(left!.value(), right!.value());
  }

  override function doPilot() {
    x <- copysign(left!.pilot(), right!.pilot());
  }

  override function doMove(κ:Kernel) {
    x <- copysign(left!.move(κ), right!.move(κ));
  }
  
  override function doGrad() {
    if x! == left!.get() {
      left!.grad(d!);
    } else {
      left!.grad(-d!);
    }
  }
}

/**
 * Lazy `copysign`.
 */
function copysign(x:Expression<Real>, y:Expression<Real>) ->
    Expression<Real> {
  if x.isConstant() && y.isConstant() {
    return box(copysign(x.value(), y.value()));
  } else {
    return construct<CopySign<Expression<Real>,Expression<Real>,Real>>(x, y);
  }
}

/**
 * Lazy `copysign`.
 */
function copysign(x:Real, y:Expression<Real>) -> Expression<Real> {
  if y.isConstant() {
    return box(copysign(x, y.value()));
  } else {
    return copysign(box(x), y);
  }
}

/**
 * Lazy `lbeta`.
 */
function copysign(x:Expression<Real>, y:Real) -> Expression<Real> {
  if x.isConstant() {
    return box(copysign(x.value(), y));
  } else {
    return copysign(x, box(y));
  }
}
