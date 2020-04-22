/**
 * Lazy `acos`.
 */
final class Acos<Argument,Value>(x:Expression<Argument>) <
    UnaryExpression<Argument,Value>(x) {
  override function computeValue(x:Argument) -> Value {
    return acos(x);
  }

  override function computeGrad(d:Value, x:Argument) -> Argument {
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
