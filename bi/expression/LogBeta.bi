/*
 * Lazy `pow`.
 */
final class LogBeta<Left,Right,Value>(x:Expression<Left>,
    y:Expression<Right>) < Expression<Value> {  
  /**
   * Base.
   */
  x:Expression<Left> <- x;
  
  /**
   * Exponent.
   */
  y:Expression<Right> <- y;

  function value() -> Value {
    return lbeta(x.value(), y.value());
  }

  function pilot() -> Value {
    return lbeta(x.pilot(), y.pilot());
  }
  
  function grad(d:Value) {
    ///@todo
  }
}

function lbeta(x:Expression<Real>, y:Expression<Real>) ->
    LogBeta<Real,Real,Real> {
  m:LogBeta<Real,Real,Real>(x, y);
  return m;
}

function lbeta(x:Real, y:Expression<Real>) -> LogBeta<Real,Real,Real> {
  return lbeta(Boxed(x), y);
}

function lbeta(x:Expression<Real>, y:Real) -> LogBeta<Real,Real,Real> {
  return lbeta(x, Boxed(y));
}
