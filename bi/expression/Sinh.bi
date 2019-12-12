/**
 * Lazy `sinh`.
 */
final class Sinh<Argument,Value>(x:Expression<Argument>) <
    UnaryExpression<Argument,Value>(x) {  
  function graft(child:Delay?) -> Expression<Value> {
    return sinh(single.graft(child));
  }

  function doValue(x:Argument) -> Value {
    return sinh(x);
  }

  function doGradient(d:Value, x:Argument) -> Argument {
    return d*cosh(x);
  }
}

/**
 * Lazy `sinh`.
 */
function sinh(x:Expression<Real>) -> Sinh<Real,Real> {
  m:Sinh<Real,Real>(x);
  return m;
}
