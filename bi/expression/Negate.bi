/*
 * Lazy `-` unary operator.
 */
final class Negate<Value>(x:Expression<Value>) < Expression<Value> {  
  /**
   * Argument.
   */
  x:Expression<Value> <- x;

  function value() -> Value {
    return -x.value();
  }

  function pilot() -> Value {
    return -x.value();
  }

  function grad(d:Value) {
    x.grad(-d);
  }
}

operator (-x:Expression<Real>) -> Negate<Real> {
  m:Negate<Real>(x);
  return m;
}

operator (-x:Expression<Integer>) -> Negate<Integer> {
  m:Negate<Integer>(x);
  return m;
}
