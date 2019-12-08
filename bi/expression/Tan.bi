/**
 * Lazy `tan`.
 */
final class Tan<Argument,Value>(x:Expression<Argument>) <
    UnaryExpression<Argument,Value>(x) {  
  function doValue(x:Argument) -> Value {
    return tan(x);
  }

  function doGradient(d:Value, x:Argument) -> Argument {
    return d*(1.0 + pow(tan(x), 2.0));
  }
}

/**
 * Lazy `tan`.
 */
function tan(x:Expression<Real>) -> Tan<Real,Real> {
  m:Tan<Real,Real>(x);
  return m;
}
