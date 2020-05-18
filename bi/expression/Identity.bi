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

  override function doPrior() -> Expression<Real>? {
    return nil;
  }

  override function doZip(x:DelayExpression, κ:Kernel) -> Real {
    assert Identity<Value>?(x)?;
    return 0.0;
  }

  override function doClearZip() {
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
