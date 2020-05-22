/**
 * Unary expression.
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

  final override function doValue() {
    x <- computeValue(single.value());
  }

  final override function doMakeConstant() {
    single.makeConstant();
  }
  
  final override function doPilot() {
    x <- computeValue(single.pilot());
  }

  final override function doRestoreCount() {
    single.restoreCount();
  }

  final override function doMove(κ:Kernel) {
    x <- computeValue(single.move(κ));
  }
  
  final override function doGrad() {
    single.grad(computeGrad(dfdx!, single.get()));
  }

  final override function doPrior(vars:RaggedArray<DelayExpression>) ->
      Expression<Real>? {
    return single.prior(vars);
  }

  /**
   * Evaluate a value.
   *
   * - x: Argument.
   *
   * Returns: Value for the given argument.
   */
  abstract function computeValue(x:Argument) -> Value;

  /**
   * Evaluate a gradient.
   *
   * - d: Outer gradient.
   * - r: Argument.
   *
   * Returns: Gradient with respect to the argument at the given position.
   */
  abstract function computeGrad(d:Value, x:Argument) -> Argument;
}
