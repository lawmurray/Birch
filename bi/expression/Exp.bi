/*
 * Lazy `exp`.
 */
final class Exp<Value>(x:Expression<Value>) < Expression<Value> {  
  /**
   * Argument.
   */
  x:Expression<Value> <- x;

  function value() -> Value {
    return exp(x.value());
  }

  function pilot() -> Value {
    return exp(x.pilot());
  }
}

function exp(x:Expression<Real>) -> Exp<Real> {
  m:Exp<Real>(x);
  return m;
}
