/**
 * Lazy `transpose`.
 */
final class MatrixTranspose<Argument,Value>(x:Expression<Argument>) <
    UnaryExpression<Argument,Value>(x) {
  function rows() -> Integer {
    return single.columns();
  }
  
  function columns() -> Integer {
    return single.rows();
  }

  function doValue(x:Argument) -> Value {
    return transpose(x);
  }

  function doGradient(d:Value, x:Argument) -> Argument {
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
