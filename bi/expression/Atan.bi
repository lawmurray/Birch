/**
 * Lazy `atan`.
 */
final class Atan<Argument,Value>(x:Expression<Argument>) <
    UnaryExpression<Argument,Value>(x) {  
  function doValue(x:Argument) -> Value {
    return atan(x);
  }

  function doGradient(d:Value, x:Argument) -> Argument {
    return d/(1.0 + x*x);
  }
}

/**
 * Lazy `atan`.
 */
function atan(x:Expression<Real>) -> Atan<Real,Real> {
  m:Atan<Real,Real>(x);
  return m;
}
