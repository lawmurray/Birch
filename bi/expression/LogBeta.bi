/**
 * Lazy `lbeta`.
 */
final class LogBeta<Left,Right,Value>(left:Expression<Left>,
    right:Expression<Right>) < BinaryExpression<Left,Right,Value>(left, right) {  
  function graft(child:Delay?) -> Expression<Value> {
    return lbeta(left.graft(child), right.graft(child));
  }

  function doValue(l:Left, r:Right) -> Value {
    return lbeta(l, r);
  }
  
  function doGradient(d:Value, l:Left, r:Right) -> (Left, Right) {
    auto d1 <- digamma(l);
    auto d2 <- digamma(r);
    auto d3 <- digamma(l + r);
    return (d*(d1 + d3), d*(d2 + d3));
  }
}

/**
 * Lazy `lbeta`.
 */
function lbeta(x:Expression<Real>, y:Expression<Real>) ->
    LogBeta<Real,Real,Real> {
  m:LogBeta<Real,Real,Real>(x, y);
  return m;
}

/**
 * Lazy `lbeta`.
 */
function lbeta(x:Real, y:Expression<Real>) -> LogBeta<Real,Real,Real> {
  return lbeta(Boxed(x), y);
}

/**
 * Lazy `lbeta`.
 */
function lbeta(x:Expression<Real>, y:Real) -> LogBeta<Real,Real,Real> {
  return lbeta(x, Boxed(y));
}
