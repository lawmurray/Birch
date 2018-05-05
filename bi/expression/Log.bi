/*
 * Lazy `log`.
 */
class Log<Value>(x:Expression<Value>) < Expression<Value> {  
  /**
   * Argument.
   */
  x:Expression<Value> <- x;

  function value() -> Value {
    return log(x.value());
  }
}

function log(x:Expression<Real>) -> Log<Real> {
  m:Log<Real>(x);
  return m;
}
