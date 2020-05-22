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

  override function doMakeConstant() {
    //
  }

  override function doPilot() {
    x <- identity(n);
  }

  override function doRestoreCount() {
    //
  }

  override function doMove(κ:Kernel) {
    //
  }

  override function doGrad() {
    //
  }

  override function doPrior(vars:RaggedArray<DelayExpression>) ->
      Expression<Real>? {
    return nil;
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
