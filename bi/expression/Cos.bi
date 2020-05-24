/**
 * Lazy `cos`.
 */
final class Cos<Argument,Value>(x:Expression<Argument>) <
    UnaryExpression<Argument,Value>(x) {
  override function computeValue(x:Argument) -> Value {
    return cos(x);
  }

  override function computeGrad(d:Value, x:Argument) -> Argument {
    return -d*sin(x);
  }
}

/**
 * Lazy `cos`.
 */
function cos(x:Expression<Real>) -> Expression<Real> {
  if x.isConstant() {
    return box(cos(x.value()));
  } else {
    m:Cos<Real,Real>(x);
    return m;
  }
}
