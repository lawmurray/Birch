/**
 * Lazy `tanh`.
 */
final class Tanh<Argument,Value>(x:Expression<Argument>) <
    UnaryExpression<Argument,Value>(x) {
  override function computeValue(x:Argument) -> Value {
    return tanh(x);
  }

  override function computeGrad(d:Value, x:Argument) -> Argument {
    return d*(1.0 + pow(tanh(x), 2.0));
  }
}

/**
 * Lazy `tanh`.
 */
function tanh(x:Expression<Real>) -> Tanh<Real,Real> {
  m:Tanh<Real,Real>(x);
  return m;
}
