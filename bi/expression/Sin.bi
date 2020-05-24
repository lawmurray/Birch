/**
 * Lazy `sin`.
 */
final class Sin<Argument,Value>(x:Expression<Argument>) <
    UnaryExpression<Argument,Value>(x) {
  override function computeValue(x:Argument) -> Value {
    return sin(x);
  }

  override function computeGrad(d:Value, x:Argument) -> Argument {
    return d*cos(x);
  }
}

/**
 * Lazy `sin`.
 */
function sin(x:Expression<Real>) -> Expression<Real> {
  if x.isConstant() {
    return box(sin(x.value()));
  } else {
    m:Sin<Real,Real>(x);
    return m;
  }
}
