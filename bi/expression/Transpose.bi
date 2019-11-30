/*
 * Lazy `transpose`.
 */
final class Transpose<Value>(x:Expression<Value>) < Expression<Value> {  
  /**
   * Argument.
   */
  x:Expression<Value> <- x;

  function value() -> Value {
    return transpose(x.value());
  }

  function pilot() -> Value {
    return transpose(x.pilot());
  }

  function grad(d:Value) {
    x.grad(transpose(d));
  }
}

function transpose(x:Expression<Real[_,_]>) -> Transpose<Real[_,_]> {
  m:Transpose<Real[_,_]>(x);
  return m;
}
