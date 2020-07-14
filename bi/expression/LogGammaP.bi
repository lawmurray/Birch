/**
 * Lazy `lgamma`.
 */
final class LogGammaP(y:Expression<Real>, z:Integer) <
    ScalarUnaryExpression<Expression<Real>,Real,Real,Real>(y) {
  /**
   * Second (fixed) argument.
   */
  z:Integer <- z;
    
  override function doEvaluate(y:Real) -> Real {
    return lgamma(y, z);
  }

  override function doEvaluateGrad(d:Real, x:Real, y:Real) -> Real {
    auto r <- 0.0;
    for i in 1..z {
      r <- r + digamma(y + 0.5*(1 - i));
    }
    return d*r;
  }
}

/**
 * Lazy `lgamma`.
 */
function lgamma(y:Expression<Real>, z:Integer) -> LogGammaP {
  return construct<LogGammaP>(y, z);
}
