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

  final function doValue() -> Value {
    return doValue(single.value());
  }
  
  final function doPilot() -> Value {
    return doValue(single.pilot());
  }
  
  final function doGrad(d:Value) {
    assert x?;
    single.grad(doGrad(d, single.get()));
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
  abstract function doGrad(d:Value, x:Argument) -> Argument;
}
