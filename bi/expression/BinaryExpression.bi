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
   * Final value.
   */
  x:Value?;
  
  /**
   * Piloted value.
   */
  x':Value?;
  
  /**
   * Proposed value.
   */
  x'':Value?;

  operator <- x:Value {
    assert !x'?;
    assert !x''?;
    this.x <- x;
  }

  final function value() -> Value {
    if !x? {
      x <- doValue(left.value(), right.value());
    }
    return x!;
  }

  final function pilot() -> Value {
    if x? {
      return x!;
    } else {
      if !x'? {
        x' <- doValue(left.pilot(), right.pilot());
      }
      return x'!;
    }
  }
  
  final function propose() -> Value {
    if x? {
      return x!;
    } else {
      if !x''? {
        x'' <- doValue(left.propose(), right.propose());
      }
      return x''!;
    }
  }
  
  final function gradPilot(d:Value) -> Boolean {
    if x? {
      return false;
    } else {
      assert x'?;
      auto l <- left.pilot();
      auto r <- right.pilot();
      dl:Left;
      dr:Right;
      (dl, dr) <- doGradient(d, l, r);
      return left.gradPilot(dl) || right.gradPilot(dr);
    }
  }

  final function gradPropose(d:Value) -> Boolean {
    if x? {
      return false;
    } else {
      assert x'?;
      auto l <- left.propose();
      auto r <- right.propose();
      dl:Left;
      dr:Right;
      (dl, dr) <- doGradient(d, l, r);
      return left.gradPropose(dl) || right.gradPropose(dr);
    }
  }

  final function ratio() -> Real {
    if x? {
      return 0.0;
    } else {
      return left.ratio() + right.ratio();
    }
  }
  
  final function accept() {
    if x? {
      // nothing to do
    } else {
      x' <- x'';
      x'' <- nil;
      left.accept();
      right.accept();
    }
  }

  final function reject() {
    if x? {
      // nothing to do
    } else {
      x'' <- nil;
      left.reject();
      right.reject();
    }
  }
  
  final function clamp() {
    if x? {
      // nothing to do
    } else {
      x <- x';
      x' <- nil;
      x'' <- nil;
      left.clamp();
      right.clamp();
    }
  }

  final function graft(child:Delay) {
    left.graft(child);
    right.graft(child);
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
