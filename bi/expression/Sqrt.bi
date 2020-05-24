/**
 * Lazy `sqrt`.
 */
final class Sqrt<Argument,Value>(x:Expression<Argument>) <
    UnaryExpression<Argument,Value>(x) {
  override function computeValue(x:Argument) -> Value {
    return sqrt(x);
  }

  override function computeGrad(d:Value, x:Argument) -> Argument {
    return d*0.5/sqrt(x);
  }
}

/**
 * Lazy `sqrt`.
 */
function sqrt(x:Expression<Real>) -> Expression<Real> {
  if x.isConstant() {
    return box(sqrt(x.value()));
  } else {
    m:Sqrt<Real,Real>(x);
    return m;
  }
}
