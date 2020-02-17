/**
 * Lazy `diagonal`.
 */
final class MatrixDiagonal<Argument,Value>(x:Expression<Argument>) <
    UnaryExpression<Argument,Value>(x) {
  function rows() -> Integer {
    return single.rows();
  }
  
  function columns() -> Integer {
    return single.columns();
  }

  function doValue(x:Argument) -> Value {
    return diagonal(x);
  }

  function doGradient(d:Value, x:Argument) -> Argument {
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
