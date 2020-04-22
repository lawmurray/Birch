/**
 * Lazy `diagonal`.
 */
final class MultivariateDiagonal<Argument,Value>(x:Expression<Argument>) <
    UnaryExpression<Argument,Value>(x) {
  function rows() -> Integer {
    return single.rows();
  }
  
  function columns() -> Integer {
    return single.rows();
  }

  function doValue(x:Argument) -> Value {
    return diagonal(x);
  }

  function doGrad(d:Value, x:Argument) -> Argument {
    return diagonal(d);
  }
}

/**
 * Lazy `diagonal`.
 */
function diagonal(x:Expression<Real[_]>) ->
    MultivariateDiagonal<Real[_],Real[_,_]> {
  m:MultivariateDiagonal<Real[_],Real[_,_]>(x);
  return m;
}
