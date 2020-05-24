/**
 * Lazy `cosh`.
 */
final class Cosh<Argument,Value>(x:Expression<Argument>) <
    UnaryExpression<Argument,Value>(x) {
  override function computeValue(x:Argument) -> Value {
    return cosh(x);
  }

  override function computeGrad(d:Value, x:Argument) -> Argument {
    return -d*sinh(x);
  }
}

/**
 * Lazy `cosh`.
 */
function cosh(x:Expression<Real>) -> Expression<Real> {
  if x.isConstant() {
    return box(cosh(x.value()));
  } else {
    m:Cosh<Real,Real>(x);
    return m;
  }
}
