/**
 * Lazy `lbeta`.
 */
final class LogBeta(y:Expression<Real>, z:Expression<Real>) <
    ScalarBinaryExpression<Expression<Real>,Expression<Real>,Real,Real,Real,
    Real,Real>(y, z) {  
  override function doEvaluate(y:Real, z:Real) -> Real {
    return lbeta(y, z);
  }
  
  override function doEvaluateGradLeft(d:Real, x:Real, y:Real, z:Real) -> Real {
    return d*(digamma(y) + digamma(y + z));
  }

  override function doEvaluateGradRight(d:Real, x:Real, y:Real, z:Real) -> Real {
    return d*(digamma(z) + digamma(y + z));
  }
}

/**
 * Lazy `lbeta`.
 */
function lbeta(x:Expression<Real>, y:Expression<Real>) -> LogBeta {
  return construct<LogBeta>(x, y);
}

/**
 * Lazy `lbeta`.
 */
function lbeta(x:Real, y:Expression<Real>) -> LogBeta {
  return lbeta(box(x), y);
}

/**
 * Lazy `lbeta`.
 */
function lbeta(x:Expression<Real>, y:Real) -> LogBeta {
  return lbeta(x, box(y));
}
