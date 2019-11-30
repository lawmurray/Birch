/*
 * Lazy `lgamma`.
 */
final class LogGamma<Value>(x:Expression<Value>) < Expression<Value> {  
  /**
   * Argument.
   */
  x:Expression<Value> <- x;

  function value() -> Value {
    return lgamma(x.value());
  }

  function pilot() -> Value {
    return lgamma(x.pilot());
  }

  function grad(d:Value) {
    ///@todo
  }
}

function lgamma(x:Expression<Real>) -> LogGamma<Real> {
  m:LogGamma<Real>(x);
  return m;
}
