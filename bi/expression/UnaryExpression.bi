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
   * Memoized value.
   */
  x:Value?;

  final function value() -> Value {
    if !x? {
      x <- doValue(single.value());
    }
    return x!;
  }
  
  final function grad(d:Value) {
    single.grad(doGradient(d, single.value()));
  } 

  final function propose() -> Value {
    x <- doValue(single.propose());
    return x!;
  }
  
  final function ratio() -> Real {
    return single.ratio();
  }
  
  final function accept() {
    single.accept();
  }

  final function reject() {
    single.reject();
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
