/**
 * Lazy `sin`.
 */
final class Sin<Argument,Value>(x:Expression<Argument>) <
    UnaryExpression<Argument,Value>(x) {  
  function graft(child:Delay) -> Expression<Value> {
    return sin(single.graft(child));
  }

  function doValue(x:Argument) -> Value {
    return sin(x);
  }

  function doGradient(d:Value, x:Argument) -> Argument {
    return d*cos(x);
  }
}

/**
 * Lazy `sin`.
 */
function sin(x:Expression<Real>) -> Sin<Real,Real> {
  m:Sin<Real,Real>(x);
  return m;
}
