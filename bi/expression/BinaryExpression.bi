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
   * Memoized result of `pilot()`.
   */
  xstar:Value?;

  /**
   * Memoized result of `propose()`.
   */
  xprime:Value?;

  final function value() -> Value {
    return doValue(left.value(), right.value());
  }

  final function pilot() -> Value {
    if !xstar? {
      xstar <- doValue(left.pilot(), right.pilot());
    }
    return xstar!;
  }

  final function propose() -> Value {
    if !xprime? {
      xprime <- doValue(left.propose(), right.propose());
    }
    return xprime!;
  }
  
  final function dpilot(d:Value) {
    auto l <- left.pilot();
    auto r <- right.pilot();
    dl:Left;
    dr:Right;
    (dl, dr) <- doGradient(d, l, r);
    
    left.dpilot(dl);
    right.dpilot(dr);
  } 

  final function dpropose(d:Value) {
    auto l <- left.propose();
    auto r <- right.propose();
    dl:Left;
    dr:Right;
    (dl, dr) <- doGradient(d, l, r);
    
    left.dpropose(dl);
    right.dpropose(dr);
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
