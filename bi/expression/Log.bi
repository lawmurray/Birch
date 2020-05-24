/**
 * Lazy `log`.
 */
final class Log<Argument,Value>(x:Expression<Argument>) <
    UnaryExpression<Argument,Value>(x) {
  override function computeValue(x:Argument) -> Value {
    return log(x);
  }

  override function computeGrad(d:Value, x:Argument) -> Argument {
    return d/x;
  }
}

/**
 * Lazy `log`.
 */
function log(x:Expression<Real>) -> Expression<Real> {
  if x.isConstant() {
    return box(log(x.value()));
  } else {
    m:Log<Real,Real>(x);
    return m;
  }
}
