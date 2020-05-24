/**
 * Lazy `lbeta`.
 */
final class LogBeta<Left,Right,Value>(left:Expression<Left>,
    right:Expression<Right>) < BinaryExpression<Left,Right,Value>(left, right) {  
  override function computeValue(l:Left, r:Right) -> Value {
    return lbeta(l, r);
  }
  
  override function computeGrad(d:Value, l:Left, r:Right) -> (Left, Right) {
    auto d1 <- digamma(l);
    auto d2 <- digamma(r);
    auto d3 <- digamma(l + r);
    return (d*(d1 + d3), d*(d2 + d3));
  }
}

/**
 * Lazy `lbeta`.
 */
function lbeta(x:Expression<Real>, y:Expression<Real>) -> Expression<Real> {
  if x.isConstant() && y.isConstant() {
    return box(lbeta(x.value(), y.value()));
  } else {
    m:LogBeta<Real,Real,Real>(x, y);
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
    return lbeta(Boxed(x), y);
  }
}

/**
 * Lazy `lbeta`.
 */
function lbeta(x:Expression<Real>, y:Real) -> Expression<Real> {
  if x.isConstant() {
    return box(lbeta(x.value(), y));
  } else {
    return lbeta(x, Boxed(y));
  }
}
