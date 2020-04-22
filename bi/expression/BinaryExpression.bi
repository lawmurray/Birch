/**
 * Abstract lazy binary expression.
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
    x <- computeValue(left.value(), right.value());
  }

  final override function doPilot() {
    x <- computeValue(left.pilot(), right.pilot());
  }
  
  final override function doGrad(d:Value) {
    assert x?;
    auto l <- left.get();
    auto r <- right.get();
    dl:Left;
    dr:Right;
    (dl, dr) <- computeGrad(d, l, r);
    left.grad(dl);
    right.grad(dr);
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
