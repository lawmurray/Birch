/**
 * Lazy `scalar`.
 */
final class MatrixScalar<Argument,Value>(x:Expression<Argument>) <
    UnaryExpression<Argument,Value>(x) {
  override function computeValue(x:Argument) -> Value {
    return scalar(x);
  }

  override function computeGrad(d:Value, x:Argument) -> Argument {
    return [[d]];
  }
}

/**
 * Lazy `scalar`.
 */
function scalar(x:Expression<Real[_,_]>) -> Expression<Real> {
  if x.isConstant() {
    return box(scalar(x.value()));
  } else {
    m:MatrixScalar<Real[_,_],Real>(x);
    return m;
  }
}
