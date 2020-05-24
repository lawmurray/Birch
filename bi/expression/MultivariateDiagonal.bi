/**
 * Lazy `diagonal`.
 */
final class MultivariateDiagonal<Argument,Value>(x:Expression<Argument>) <
    UnaryExpression<Argument,Value>(x) {
  override function rows() -> Integer {
    return single.rows();
  }
  
  override function columns() -> Integer {
    return single.rows();
  }

  override function computeValue(x:Argument) -> Value {
    return diagonal(x);
  }

  override function computeGrad(d:Value, x:Argument) -> Argument {
    return diagonal(d);
  }
}

/**
 * Lazy `diagonal`.
 */
function diagonal(x:Expression<Real[_]>) -> Expression<Real[_,_]> {
  if x.isConstant() {
    return box(matrix(diagonal(x.value())));
  } else {
    m:MultivariateDiagonal<Real[_],Real[_,_]>(x);
    return m;
  }
}
