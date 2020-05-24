/**
 * Lazy `acos`.
 */
final class Acos<Argument,Value>(x:Expression<Argument>) <
    UnaryExpression<Argument,Value>(x) {
  override function computeValue(x:Argument) -> Value {
    return acos(x);
  }

  override function computeGrad(d:Value, x:Argument) -> Argument {
    return -d/sqrt(1.0 - x*x);
  }
}

/**
 * Lazy `acos`.
 */
function acos(x:Expression<Real>) -> Expression<Real> {
  if x.isConstant() {
    return box(acos(x.value()));
  } else {
    m:Acos<Real,Real>(x);
    return m;
  }
}
