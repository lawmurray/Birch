/**
 * Lazy `diagonal`.
 */
final class MatrixDiagonal<Argument,Value>(x:Expression<Argument>) <
    UnaryExpression<Argument,Value>(x) {
  override function rows() -> Integer {
    return single.rows();
  }
  
  override function columns() -> Integer {
    return single.columns();
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
function diagonal(x:Expression<Real[_,_]>) ->
    MatrixDiagonal<Real[_,_],Real[_]> {
  m:MatrixDiagonal<Real[_,_],Real[_]>(x);
  return m;
}
