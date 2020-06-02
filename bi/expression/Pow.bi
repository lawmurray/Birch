/**
 * Lazy `pow`.
 */
final class Pow<Left,Right,Value>(left:Expression<Left>,
    right:Expression<Right>) < BinaryExpression<Left,Right,Value>(left, right) {  
  override function computeValue(l:Left, r:Right) -> Value {
    return pow(l, r);
  }

  override function computeGrad(d:Value, l:Left, r:Right) -> (Left, Right) {
    auto dl <- d*r*pow(l, r - 1);
    auto dr <- 0.0;
    if l > 0.0 {
      dr <- d*pow(l, r)*log(l);
    }
    return (dl, dr);
  }
}

/**
 * Lazy `pow`.
 */
function pow(x:Expression<Real>, y:Expression<Real>) -> Expression<Real> {
  if x.isConstant() && y.isConstant() {
    return box(pow(x.value(), y.value()));
  } else {
    m:Pow<Real,Real,Real>(x, y);
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
