/**
 * Lazy `pow`.
 */
final class Pow<Left,Right,Value>(left:Expression<Left>,
    right:Expression<Right>) < BinaryExpression<Left,Right,Value>(left, right) {  
  function graft(child:Delay?) -> Expression<Value> {
    return pow(left.graft(child), right.graft(child));
  }

  function doValue(l:Left, r:Right) -> Value {
    return pow(l, r);
  }

  function doGradient(d:Value, l:Left, r:Right) -> (Left, Right) {
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
function pow(x:Expression<Real>, y:Expression<Real>) -> Pow<Real,Real,Real> {
  m:Pow<Real,Real,Real>(x, y);
  return m;
}

/**
 * Lazy `pow`.
 */
function pow(x:Real, y:Expression<Real>) -> Pow<Real,Real,Real> {
  return pow(Boxed(x), y);
}

/**
 * Lazy `pow`.
 */
function pow(x:Expression<Real>, y:Real) -> Pow<Real,Real,Real> {
  return pow(x, Boxed(y));
}
