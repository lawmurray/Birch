/**
 * Lazy `transpose`.
 */
final class Transpose<Argument,Value>(x:Expression<Argument>) <
    UnaryExpression<Argument,Value>(x) {
  function rows() -> Integer {
    return single.columns();
  }
  
  function columns() -> Integer {
    return single.rows();
  }

  function graft() -> Expression<Value> {
    return transpose(single.graft());
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
function transpose(x:Expression<Real[_,_]>) -> Transpose<Real[_,_],Real[_,_]> {
  m:Transpose<Real[_,_],Real[_,_]>(x);
  return m;
}
