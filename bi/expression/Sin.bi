/**
 * Lazy `sin`.
 */
final class Sin<Argument,Value>(x:Expression<Argument>) <
    UnaryExpression<Argument,Value>(x) {
  override function computeValue(x:Argument) -> Value {
    return sin(x);
  }

  override function computeGrad(d:Value, x:Argument) -> Argument {
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
