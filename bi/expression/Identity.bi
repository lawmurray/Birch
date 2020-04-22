/**
 * Lazy `identity`.
 */
final class Identity<Value>(n:Integer) < Expression<Value> {
  /**
   * Size.
   */
  n:Integer <- n;
  
  function rows() -> Integer {
    return n;
  }
  
  function columns() -> Integer {
    return n;
  }

  function doValue() -> Value {
    return identity(n);
  }

  function doPilot() -> Value {
    return identity(n);
  }

  function doGrad(d:Value) {
    //
  }
}

/**
 * Lazy `identity`. This is named `Identity` rather than `identity` to
 * distinguish from the eager version.
 */
function Identity(n:Integer) -> Identity<Real[_,_]> {
  m:Identity<Real[_,_]>(n);
  return m;
}
