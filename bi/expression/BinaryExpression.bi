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

  final function setChild(child:Delay) {
    left.setChild(child);
    right.setChild(child);
  }

  final function value() -> Value {
    if !x? {
      x <- doValue(left.value(), right.value());
    }
    return x!;
  }
  
  final function grad(d:Value) -> Boolean {
    auto l <- left.value();
    auto r <- right.value();
    dl:Left;
    dr:Right;
    (dl, dr) <- doGradient(d, l, r);
    auto leftGrad <- left.grad(dl);
    auto rightGrad <- right.grad(dr);
    return leftGrad || rightGrad;  // done this way to avoid short circuit
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
   * - d: Outer gradient.
   * - l: Left argument.
   * - r: Right argument.
   *
   * Returns: Gradient with respect to the left and right arguments at the
   * given position.
   */
  abstract function doGradient(d:Value, l:Left, r:Right) -> (Left, Right);
}
