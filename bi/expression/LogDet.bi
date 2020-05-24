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
function ldet(x:Expression<LLT>) -> Expression<Real> {
  if x.isConstant() {
    return box(ldet(x.value()));
  } else {
    m:LogDet<LLT,Real>(x);
    return m;
  }
}
