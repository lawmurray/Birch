/**
 * Lazy `scalar`.
 */
final class MultivariateScalar<Argument,Value>(x:Expression<Argument>) <
    UnaryExpression<Argument,Value>(x) {
  override function computeValue(x:Argument) -> Value {
    return scalar(x);
  }

  override function computeGrad(d:Value, x:Argument) -> Argument {
    return [d];
  }
}

/**
 * Lazy `scalar`.
 */
function scalar(x:Expression<Real[_]>) -> Expression<Real> {
  if x.isConstant() {
    return box(scalar(x.value()));
  } else {
    m:MultivariateScalar<Real[_],Real>(x);
    return m;
  }
}
