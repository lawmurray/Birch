/*
 * Lazy `sqrt`.
 */
final class Sqrt<Value>(x:Expression<Value>) < Expression<Value> {  
  /**
   * Argument.
   */
  x:Expression<Value> <- x;

  function value() -> Value {
    return sqrt(x.value());
  }

  function pilot() -> Value {
    return sqrt(x.pilot());
  }

  function grad(d:Value) {
    x.grad(d*0.5/sqrt(x.pilot()));
  }
}

function sqrt(x:Expression<Real>) -> Sqrt<Real> {
  m:Sqrt<Real>(x);
  return m;
}
