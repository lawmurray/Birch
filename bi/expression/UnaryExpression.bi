/**
 * Abstract lazy unary expression.
 *
 * - Argument: Argument type.
 * - Value: Value type.
 */
abstract class UnaryExpression<Argument,Value>(single:Expression<Argument>) <
    Expression<Value> {  
  /**
   * Final value.
   */
  x:Value?;

  /**
   * Single argument.
   */
  single:Expression<Argument> <- single;

  operator <- x:Value {
    this.x <- x;
  }

  final function setChild(child:Delay) {
    single.setChild(child);
  }

  final function value() -> Value {
    if !x? {
      x <- doValue(single.value());
    }
    return x!;
  }
  
  final function grad(d:Value) {
    assert x?;
    single.grad(doGradient(d, single));
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
