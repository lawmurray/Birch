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
  
  /**
   * Memoized value.
   */
  x:Value?;

  final function value() -> Value {
    if !x? {
      x <- doValue(left.value(), right.value());
    }
    return x!;
  }
  
  final function grad(d:Value) {
    auto l <- left.value();
    auto r <- right.value();
    dl:Left;
    dr:Right;
    (dl, dr) <- doGradient(d, l, r);
    
    left.grad(dl);
    right.grad(dr);
  }

  final function propose() -> Value {
    x <- doValue(left.propose(), right.propose());
    return x!;
  }

  final function ratio() -> Real {
    return left.ratio() + right.ratio();
  }
  
  final function accept() {
    left.accept();
    right.accept();
  }

  final function reject() {
    left.reject();
    right.reject();
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
