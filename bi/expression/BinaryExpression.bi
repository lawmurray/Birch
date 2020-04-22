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
   * Value.
   */
  x:Value?;

  /**
   * Left argument.
   */
  left:Expression<Left> <- left;
  
  /**
   * Right argument.
   */
  right:Expression<Right> <- right;

  operator <- x:Value {
    this.x <- x;
  }

  final function get() -> Value {
    return x!;
  }

  final function value() -> Value {
    if !x? {
      x <- doValue(left.value(), right.value());
    }
    return x!;
  }

  final function pilot() -> Value {
    if !x? {
      x <- doValue(left.pilot(), right.pilot());
    }
    return x!;
  }
  
  final function grad(d:Value) {
    assert x?;
    auto l <- left.get();
    auto r <- right.get();
    dl:Left;
    dr:Right;
    (dl, dr) <- doGradient(d, l, r);
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
  abstract function doGradient(d:Value, l:Left, r:Right) -> (Left, Right);
}
