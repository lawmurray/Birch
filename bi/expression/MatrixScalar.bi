/**
 * Lazy `scalar`.
 */
final class MatrixScalar<Argument,Value>(x:Expression<Argument>) <
    UnaryExpression<Argument,Value>(x) {  
  function graft(child:Delay) -> Expression<Value> {
    return scalar(single.graft(child));
  }
  
  function doValue(x:Argument) -> Value {
    return scalar(x);
  }

  function doGradient(d:Value, x:Argument) -> Argument {
    return [[d]];
  }
}

/**
 * Lazy `scalar`.
 */
function scalar(x:Expression<Real[_,_]>) -> MatrixScalar<Real[_,_],Real> {
  m:MatrixScalar<Real[_,_],Real>(x);
  return m;
}
