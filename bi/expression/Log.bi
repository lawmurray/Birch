/*
 * Lazy `log`.
 */
final class Log<Value>(x:Expression<Value>) < Expression<Value> {  
  /**
   * Argument.
   */
  x:Expression<Value> <- x;

  function value() -> Value {
    return log(x.value());
  }

  function pilot() -> Value {
    return log(x.pilot());
  }

  function grad(d:Value) {
    x.grad(d/x.pilot());
  }
}

function log(x:Expression<Real>) -> Log<Real> {
  m:Log<Real>(x);
  return m;
}
