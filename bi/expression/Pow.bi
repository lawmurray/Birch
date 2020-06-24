/**
 * Lazy `pow`.
 */
final class Pow<Left,Right,Value>(left:Left, right:Right) <
    ScalarBinaryExpression<Left,Right,Value>(left, right) {  
  override function doValue() {
    x <- pow(left!.value(), right!.value());
  }

  override function doPilot() {
    x <- pow(left!.pilot(), right!.pilot());
  }

  override function doMove(κ:Kernel) {
    x <- pow(left!.move(κ), right!.move(κ));
  }

  override function doGrad() {
    auto l <- left!.get();
    auto r <- right!.get();
    auto dl <- d!*r*pow(l, r - 1.0);
    auto dr <- 0.0;
    if l > 0.0 {
      dr <- d!*pow(l, r)*log(l);
    }
    left!.grad(dl);
    right!.grad(dr);
  }
}

/**
 * Lazy `pow`.
 */
function pow(x:Expression<Real>, y:Expression<Real>) -> Expression<Real> {
  if x.isConstant() && y.isConstant() {
    return box(pow(x.value(), y.value()));
  } else {
    m:Pow<Expression<Real>,Expression<Real>,Real>(x, y);
    return m;
  }
}

/**
 * Lazy `pow`.
 */
function pow(x:Real, y:Expression<Real>) -> Expression<Real> {
  if y.isConstant() {
    return box(pow(x, y.value()));
  } else {
    return pow(box(x), y);
  }
}

/**
 * Lazy `pow`.
 */
function pow(x:Expression<Real>, y:Real) -> Expression<Real> {
  if x.isConstant() {
    return box(pow(x.value(), y));
  } else {
    return pow(x, box(y));
  }
}
