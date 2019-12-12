/**
 * Lazy `scalar`.
 */
final class MultivariateScalar<Argument,Value>(x:Expression<Argument>) <
    UnaryExpression<Argument,Value>(x) {  
  function graft() -> Expression<Value> {
    return scalar(single.graft());
  }
  
  function doValue(x:Argument) -> Value {
    return scalar(x);
  }

  function doGradient(d:Value, x:Argument) -> Argument {
    return [d];
  }
}

/**
 * Lazy `scalar`.
 */
function scalar(x:Expression<Real[_]>) -> MultivariateScalar<Real[_],Real> {
  m:MultivariateScalar<Real[_],Real>(x);
  return m;
}
