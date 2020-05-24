/**
 * Lazy `atan`.
 */
final class Atan<Argument,Value>(x:Expression<Argument>) <
    UnaryExpression<Argument,Value>(x) {
  override function computeValue(x:Argument) -> Value {
    return atan(x);
  }

  override function computeGrad(d:Value, x:Argument) -> Argument {
    return d/(1.0 + x*x);
  }
}

/**
 * Lazy `atan`.
 */
function atan(x:Expression<Real>) -> Expression<Real> {
  if x.isConstant() {
    return box(atan(x.value()));
  } else {
    m:Atan<Real,Real>(x);
    return m;
  }
}
