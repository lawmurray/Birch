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

  function get() -> Value {
    return identity(n);
  }

  function value() -> Value {
    return identity(n);
  }

  function pilot() -> Value {
    return identity(n);
  }

  function grad(d:Value) {
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
