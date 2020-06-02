/**
 * Lazy `log1p`.
 */
final class Log1p<Argument,Value>(x:Expression<Argument>) <
    UnaryExpression<Argument,Value>(x) {
  override function computeValue(x:Argument) -> Value {
    return log1p(x);
  }

  override function computeGrad(d:Value, x:Argument) -> Argument {
    return d/(1.0 + x);
  }
}

/**
 * Lazy `log1p`.
 */
function log1p(x:Expression<Real>) -> Expression<Real> {
  if x.isConstant() {
    return box(log1p(x.value()));
  } else {
    m:Log1p<Real,Real>(x);
    return m;
  }
}
