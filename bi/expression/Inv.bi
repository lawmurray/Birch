/*
 * Lazy `inv`.
 */
final class Inv<Value>(x:Expression<Value>) < Expression<Value> {  
  /**
   * Argument.
   */
  x:Expression<Value> <- x;

  function value() -> Value {
    return inv(x.value());
  }

  function pilot() -> Value {
    return inv(x.pilot());
  }

  function grad(d:Value) {
    ///@todo
  }
}

function inv(x:Expression<Real[_,_]>) -> Inv<Real[_,_]> {
  m:Inv<Real[_,_]>(x);
  return m;
}
