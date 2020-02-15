/**
 * Lazy `transpose`.
 */
final class MultivariateTranspose<Argument,Value>(x:Expression<Argument>) <
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
    return vector(d);
  }
}

/**
 * Lazy `transpose`.
 */
function transpose(x:Expression<Real[_]>) ->
    MultivariateTranspose<Real[_],Real[_,_]> {
  m:MultivariateTranspose<Real[_],Real[_,_]>(x);
  return m;
}
