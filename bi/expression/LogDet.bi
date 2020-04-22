/**
 * Lazy `ldet`.
 */
final class LogDet<Argument,Value>(x:Expression<Argument>) <
    UnaryExpression<Argument,Value>(x) {
  override function computeValue(x:Argument) -> Value {
    return ldet(x);
  }

  override function computeGrad(d:Value, x:Argument) -> Argument {
    ///@todo
    assert false;
  }
}

/**
 * Lazy `ldet`.
 */
function ldet(x:Expression<LLT>) -> LogDet<LLT,Real> {
  m:LogDet<LLT,Real>(x);
  return m;
}
