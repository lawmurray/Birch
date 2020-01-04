/**
 * Lazy `cos`.
 */
final class Cos<Argument,Value>(x:Expression<Argument>) <
    UnaryExpression<Argument,Value>(x) {
  function doValue(x:Argument) -> Value {
    return cos(x);
  }

  function doGradient(d:Value, x:Argument) -> Argument {
    return -d*sin(x);
  }
}

/**
 * Lazy `cos`.
 */
function cos(x:Expression<Real>) -> Cos<Real,Real> {
  m:Cos<Real,Real>(x);
  return m;
}
