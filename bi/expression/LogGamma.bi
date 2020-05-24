/**
 * Lazy `lgamma`.
 */
final class LogGamma<Argument,Value>(x:Expression<Argument>) <
    UnaryExpression<Argument,Value>(x) {
  override function computeValue(x:Argument) -> Value {
    return lgamma(x);
  }

  override function computeGrad(d:Value, x:Argument) -> Argument {
    return d*digamma(x);
  }
}

/**
 * Lazy `lgamma`.
 */
function lgamma(x:Expression<Real>) -> Expression<Real> {
  if x.isConstant() {
    return box(lgamma(x.value()));
  } else {
    m:LogGamma<Real,Real>(x);
    return m;
  }
}
