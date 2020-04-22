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

  final function doValue() -> Value {
    return doValue(left.value(), right.value());
  }

  final function doPilot() -> Value {
    return doValue(left.pilot(), right.pilot());
  }
  
  final function doGrad(d:Value) {
    assert x?;
    auto l <- left.get();
    auto r <- right.get();
    dl:Left;
    dr:Right;
    (dl, dr) <- doGrad(d, l, r);
    left.grad(dl);
    right.grad(dr);
  }

  /**
   * Evaluate a value.
   *
   * - l: Left argument.
   * - r: Right argument.
   *
   * Returns: Value for the given arguments.
   */
  abstract function doValue(l:Left, r:Right) -> Value;

  /**
   * Evaluate a gradient.
   *
   * - d: Upstream gradient.
   * - l: Left argument.
   * - r: Right argument.
   *
   * Returns: Gradient with respect to the left and right arguments at the
   * given position.
   */
  abstract function doGrad(d:Value, l:Left, r:Right) -> (Left, Right);
}
