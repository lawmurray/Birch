/**
 * Lazy `exp`.
 */
final class Exp<Argument,Value>(x:Expression<Argument>) <
    UnaryExpression<Argument,Value>(x) {
  function graft() -> Expression<Value> {
    return exp(single.graft());
  }

  function doValue(x:Argument) -> Value {
    return exp(x);
  }

  function doGradient(d:Value, x:Argument) -> Argument {
    return d*exp(x);
  }
}

/**
 * Lazy `exp`.
 */
function exp(x:Expression<Real>) -> Exp<Real,Real> {
  m:Exp<Real,Real>(x);
  return m;
}
