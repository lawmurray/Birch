/**
 * Lazy `identity`.
 */
final class Identity<Value>(n:Integer) < Expression<Value> {
  /**
   * Size.
   */
  n:Integer <- n;
  
  override function rows() -> Integer {
    return n;
  }
  
  override function columns() -> Integer {
    return n;
  }

  override function doValue() {
    x <- identity(n);
  }

  override function doPilot() {
    x <- identity(n);
  }

  override function doGrad(d:Value) {
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
