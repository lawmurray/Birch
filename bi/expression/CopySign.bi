/**
 * Lazy `copysign`.
 */
final class CopySign<Left,Right,Value>(left:Expression<Left>,
    right:Expression<Right>) < BinaryExpression<Left,Right,Value>(left, right) {  
  override function computeValue(l:Left, r:Right) -> Value {
    return copysign(l, r);
  }
  
  override function computeGrad(d:Value, l:Left, r:Right) -> (Left, Right) {
    if copysign(l, r) == l {
      return (d, 0.0);
    } else {
      return (-d, 0.0);
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
    m:CopySign<Real,Real,Real>(x, y);
    return m;
  }
}

/**
 * Lazy `copysign`.
 */
function copysign(x:Real, y:Expression<Real>) -> Expression<Real> {
  if y.isConstant() {
    return box(copysign(x, y.value()));
  } else {
    return copysign(Boxed(x), y);
  }
}

/**
 * Lazy `lbeta`.
 */
function copysign(x:Expression<Real>, y:Real) -> Expression<Real> {
  if x.isConstant() {
    return box(copysign(x.value(), y));
  } else {
    return copysign(x, Boxed(y));
  }
}
