/**
 * Lazy `tan`.
 */
final class Tan<Argument,Value>(x:Expression<Argument>) <
    UnaryExpression<Argument,Value>(x) {
  override function computeValue(x:Argument) -> Value {
    return tan(x);
  }

  override function computeGrad(d:Value, x:Argument) -> Argument {
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
