/**
 * Lazy `log`.
 */
final class Log<Argument,Value>(x:Expression<Argument>) <
    UnaryExpression<Argument,Value>(x) {  
  function graft() -> Expression<Value> {
    return log(single.graft());
  }

  function doValue(x:Argument) -> Value {
    return log(x);
  }

  function doGradient(d:Value, x:Argument) -> Argument {
    return d/x;
  }
}

/**
 * Lazy `log`.
 */
function log(x:Expression<Real>) -> Log<Real,Real> {
  m:Log<Real,Real>(x);
  return m;
}
