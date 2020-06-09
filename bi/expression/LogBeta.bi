/**
 * Lazy `lbeta`.
 */
final class LogBeta<Left,Right,Value>(left:Left, right:Right) <
    ScalarBinaryExpression<Left,Right,Value>(left, right) {  
  override function doValue() {
    x <- lbeta(left.value(), right.value());
  }

  override function doGet() {
    x <- lbeta(left.get(), right.get());
  }

  override function doPilot() {
    x <- lbeta(left.pilot(), right.pilot());
  }

  override function doMove(κ:Kernel) {
    x <- lbeta(left.move(κ), right.move(κ));
  }
  
  override function doGrad() {
    auto l <- left.get();
    auto r <- right.get();
    auto d1 <- digamma(l);
    auto d2 <- digamma(r);
    auto d3 <- digamma(l + r);
    left.grad(d!*(d1 + d3));
    right.grad(d!*(d2 + d3));
  }
}

/**
 * Lazy `lbeta`.
 */
function lbeta(x:Expression<Real>, y:Expression<Real>) -> Expression<Real> {
  if x.isConstant() && y.isConstant() {
    return box(lbeta(x.value(), y.value()));
  } else {
    m:LogBeta<Expression<Real>,Expression<Real>,Real>(x, y);
    return m;
  }
}

/**
 * Lazy `lbeta`.
 */
function lbeta(x:Real, y:Expression<Real>) -> Expression<Real> {
  if y.isConstant() {
    return box(lbeta(x, y.value()));
  } else {
    return lbeta(box(x), y);
  }
}

/**
 * Lazy `lbeta`.
 */
function lbeta(x:Expression<Real>, y:Real) -> Expression<Real> {
  if x.isConstant() {
    return box(lbeta(x.value(), y));
  } else {
    return lbeta(x, box(y));
  }
}
