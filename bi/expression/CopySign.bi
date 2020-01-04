/**
 * Lazy `copysign`.
 */
final class CopySign<Left,Right,Value>(left:Expression<Left>,
    right:Expression<Right>) < BinaryExpression<Left,Right,Value>(left, right) {  
  function doValue(l:Left, r:Right) -> Value {
    return copysign(l, r);
  }
  
  function doGradient(d:Value, l:Left, r:Right) -> (Left, Right) {
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
    CopySign<Real,Real,Real> {
  m:CopySign<Real,Real,Real>(x, y);
  return m;
}

/**
 * Lazy `copysign`.
 */
function copysign(x:Real, y:Expression<Real>) -> CopySign<Real,Real,Real> {
  return copysign(Boxed(x), y);
}

/**
 * Lazy `lbeta`.
 */
function copysign(x:Expression<Real>, y:Real) -> CopySign<Real,Real,Real> {
  return copysign(x, Boxed(y));
}
