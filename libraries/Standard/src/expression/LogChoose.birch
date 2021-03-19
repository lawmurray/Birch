/**
 * Lazy `lchoose`.
 */
final class LogChoose(y:Expression<Integer>, z:Expression<Integer>) <
    ScalarBinaryExpression<Expression<Integer>,Expression<Integer>,Integer,
    Integer,Real,Real,Real>(y, z) {  
  override function doEvaluate(y:Integer, z:Integer) -> Real {
    return lchoose(y, z);
  }
  
  override function doEvaluateGradLeft(d:Real, x:Real, y:Integer, z:Integer) -> Real {
    ///@todo Can gamma function be used to provide a gradient here?
    return 0.0;
  }

  override function doEvaluateGradRight(d:Real, x:Real, y:Integer, z:Integer) -> Real {
    ///@todo Can gamma function be used to provide a gradient here?
    return 0.0;
  }
}

/**
 * Lazy `lchoose`.
 */
function lchoose(y:Expression<Integer>, z:Expression<Integer>) -> LogChoose {
  return construct<LogChoose>(y, z);
}

/**
 * Lazy `lchoose`.
 */
function lchoose(y:Integer, z:Expression<Integer>) -> LogChoose {
  return lchoose(box(y), z);
}

/**
 * Lazy `lchoose`.
 */
function lchoose(y:Expression<Integer>, z:Integer) -> LogChoose {
  return lchoose(y, box(z));
}
