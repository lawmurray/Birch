/**
 * Lazy `sinh`.
 */
final class Sinh<Argument,Value>(x:Expression<Argument>) <
    UnaryExpression<Argument,Value>(x) {
  override function computeValue(x:Argument) -> Value {
    return sinh(x);
  }

  override function computeGrad(d:Value, x:Argument) -> Argument {
    return d*cosh(x);
  }
}

/**
 * Lazy `sinh`.
 */
function sinh(x:Expression<Real>) -> Expression<Real> {
  if x.isConstant() {
    return box(sinh(x.value()));
  } else {
    m:Sinh<Real,Real>(x);
    return m;
  }
}
