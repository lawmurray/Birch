/**
 * Abstract lazy unary expression.
 *
 * - Argument: Argument type.
 * - Value: Value type.
 */
abstract class UnaryExpression<Argument,Value>(x:Expression<Argument>) <
    Expression<Value> {  
  /**
   * Argument.
   */
  x:Expression<Argument> <- x;

  /**
   * Memoized result of `pilot()`.
   */
  xstar:Value?;

  /**
   * Memoized result of `propose()`.
   */
  xprime:Value?;

  final function value() -> Value {
    return doValue(x.value());
  }

  final function pilot() -> Value {
    if !xstar? {
      xstar <- doValue(x.pilot());
    }
    return xstar!;
  }

  final function propose() -> Value {
    if !xprime? {
      xprime <- doValue(x.propose());
    }
    return xprime!;
  }
  
  final function dpilot(d:Value) {
    x.dpilot(doGradient(d, x.pilot()));
  } 

  final function dpropose(d:Value) {
    x.dpropose(doGradient(d, x.propose()));
  }

  final function ratio() -> Real {
    return x.ratio();
  }
  
  final function accept() {
    x.accept();
  }
  
  final function reject() {
    x.reject();
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
