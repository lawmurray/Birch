/**
 * Lazy `acos`.
 */
final class Acos<Argument,Value>(x:Expression<Argument>) <
    UnaryExpression<Argument,Value>(x) {
  function doValue(x:Argument) -> Value {
    return acos(x);
  }

  function doGradient(d:Value, x:Argument) -> Argument {
    return -d/sqrt(1.0 - x*x);
  }
}

/**
 * Lazy `acos`.
 */
function acos(x:Expression<Real>) -> Acos<Real,Real> {
  m:Acos<Real,Real>(x);
  return m;
}
