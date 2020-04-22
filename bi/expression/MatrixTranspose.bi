/**
 * Lazy `transpose`.
 */
final class MatrixTranspose<Argument,Value>(x:Expression<Argument>) <
    UnaryExpression<Argument,Value>(x) {
  override function rows() -> Integer {
    return single.columns();
  }
  
  override function columns() -> Integer {
    return single.rows();
  }

  override function computeValue(x:Argument) -> Value {
    return transpose(x);
  }

  override function computeGrad(d:Value, x:Argument) -> Argument {
    return transpose(d);
  }
}

/**
 * Lazy `transpose`.
 */
function transpose(x:Expression<Real[_,_]>) ->
    MatrixTranspose<Real[_,_],Real[_,_]> {
  m:MatrixTranspose<Real[_,_],Real[_,_]>(x);
  return m;
}
