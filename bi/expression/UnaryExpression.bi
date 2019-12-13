/**
 * Abstract lazy unary expression.
 *
 * - Argument: Argument type.
 * - Value: Value type.
 */
abstract class UnaryExpression<Argument,Value>(single:Expression<Argument>) <
    Expression<Value> {  
  /**
   * Single argument.
   */
  single:Expression<Argument> <- single;

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

  final function setChild(child:Delay) {
    single.setChild(child);
  }

  final function value() -> Value {
    if !x? {
      x <- doValue(single.value());
      x' <- nil;
      x'' <- nil;
    }
    return x!;
  }
  
  final function pilot() -> Value {
    if x? {
      return x!;
    } else {
      if !x'? {
        x' <- doValue(single.pilot());
      }
      return x'!;
    }
  }
  
  final function propose() -> Value {
    if x? {
      return x!;
    } else {
      if !x''? {
        x'' <- doValue(single.propose());
      }
      return x''!;
    }
  }
  
  final function gradPilot(d:Value) -> Boolean {
    if x? {
      return false;
    } else {
      assert x'?;
      return single.gradPilot(doGradient(d, single.pilot()));
    }
  }

  final function gradPropose(d:Value) -> Boolean {
    if x? {
      return false;
    } else {
      assert x''?;
      return single.gradPropose(doGradient(d, single.propose()));
    }
  }
  
  final function ratio() -> Real {
    if x? {
      return 0.0;
    } else {
      return single.ratio();
    }
  }
  
  final function accept() {
    if x? {
      // nothing to do
    } else {
      x' <- x'';
      x'' <- nil;
      single.accept();
    }
  }

  final function reject() {
    if x? {
      // nothing to do
    } else {
      x'' <- nil;
      single.reject();
    }
  }
  
  final function clamp() {
    if x? {
      // nothing to do
    } else {
      x <- x';
      x' <- nil;
      x'' <- nil;
      single.clamp();
    }
  }
  
  /**
   * Evaluate a value.
   *
   * - x: Argument.
   *
   * Returns: Value for the given argument.
   */
  abstract function doValue(x:Argument) -> Value;

  /**
   * Evaluate a gradient.
   *
   * - d: Outer gradient.
   * - r: Argument.
   *
   * Returns: Gradient with respect to the argument at the given position.
   */
  abstract function doGradient(d:Value, x:Argument) -> Argument;
}
