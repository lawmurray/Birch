/**
 * Lazy `cosh`.
 */
final class Cosh<Argument,Value>(x:Expression<Argument>) <
    UnaryExpression<Argument,Value>(x) {  
  function graft(child:Delay?) -> Expression<Value> {
    return cosh(single.graft(child));
  }

  function doValue(x:Argument) -> Value {
    return cosh(x);
  }

  function doGradient(d:Value, x:Argument) -> Argument {
    return -d*sinh(x);
  }
}

/**
 * Lazy `cosh`.
 */
function cosh(x:Expression<Real>) -> Cosh<Real,Real> {
  m:Cosh<Real,Real>(x);
  return m;
}
