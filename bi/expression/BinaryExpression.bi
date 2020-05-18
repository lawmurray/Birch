/**
 * Binary expression.
 *
 * - Left: Left argument type.
 * - Right: Right argument type.
 * - Value: Value type.
 */
abstract class BinaryExpression<Left,Right,Value>(left:Expression<Left>,
    right:Expression<Right>) < Expression<Value> {  
  /**
   * Left argument.
   */
  left:Expression<Left> <- left;
  
  /**
   * Right argument.
   */
  right:Expression<Right> <- right;

  final override function doValue() {
    auto l <- left.value();  // ensure left-to-right recursion
    auto r <- right.value();
    x <- computeValue(l, r);
  }
  
  final override function doMakeConstant() {
    left.makeConstant();
    right.makeConstant();
  }

  final override function doPilot() {
    auto l <- left.pilot();  // ensure left-to-right recursion
    auto r <- right.pilot();
    x <- computeValue(l, r);
  }

  final override function doRestoreCount() {
    left.restoreCount();
    right.restoreCount();
  }

  final override function doMove(κ:Kernel) {
    auto l <- left.move(κ);  // ensure left-to-right recursion
    auto r <- right.move(κ);
    x <- computeValue(l, r);
  }
  
  final override function doGrad() {
    auto l <- left.get();
    auto r <- right.get();
    dl:Left;
    dr:Right;
    (dl, dr) <- computeGrad(dfdx!, l, r);
    left.grad(dl);
    right.grad(dr);
  }
  
  final override function doPrior() -> Expression<Real>? {
    auto l <- left.prior();
    auto r <- right.prior();
    if l? && r? {
      return l! + r!;
    } else if l? {
      return l!;
    } else if r? {
      return r!;
    } else {
      return nil;
    }
  }

  final override function doZip(x:DelayExpression, κ:Kernel) -> Real {
    auto y <- BinaryExpression<Left,Right,Value>?(x);
    assert y?;
    auto l <- left.zip(y!.left, κ);
    auto r <- right.zip(y!.right, κ);
    return l + r;
  }
  
  final override function doClearZip() {
    left.clearZip();
    right.clearZip();
  }

  /**
   * Compute a value.
   *
   * - l: Left argument.
   * - r: Right argument.
   *
   * Returns: Value for the given arguments.
   */
  abstract function computeValue(l:Left, r:Right) -> Value;

  /**
   * Compute a gradient.
   *
   * - d: Upstream gradient.
   * - l: Left argument.
   * - r: Right argument.
   *
   * Returns: Gradient with respect to the left and right arguments at the
   * given position.
   */
  abstract function computeGrad(d:Value, l:Left, r:Right) -> (Left, Right);
}
