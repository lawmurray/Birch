/**
 * Scalar unary expression.
 *
 * - `Argument`: Argument type. Should derive from `Expression<...>`.
 * - `ArgumentValue`: Argument value type. This is the type to which the left
 *   argument evaluates.
 * - `ArgumentGradient`: Argument upstream gradient type. This is the type of
 *   the upstream gradient that the argument accepts. It should be `Real`,
 *   `Real[_]`, or `Real[_,_]`.
 * - `Value`: The type to which the expression evaluates.
 */
abstract class ScalarUnaryExpression<Argument,ArgumentValue,ArgumentGradient,
    Value>(y:Argument) < ScalarExpression<Value> {
  /**
   * Argument.
   */
  y:Argument? <- y;

  /*
   * Evaluate.
   */
  abstract function doEvaluate(y:ArgumentValue) -> Value;
  
  /*
   * Evaluate the gradient.
   */
  abstract function doEvaluateGrad(d:Real, x:Value, y:ArgumentValue) ->
      ArgumentGradient;

  final override function doDepth() -> Integer {
    return y!.depth() + 1;
  }

  final override function doValue() -> Value {
    return doEvaluate(y!.value());
  }

  final override function doPilot(gen:Integer) -> Value {
    return doEvaluate(y!.pilot(gen));
  }

  final override function doGet() -> Value {
    return doEvaluate(y!.get());
  }

  final override function doMove(gen:Integer, κ:Kernel) -> Value {
    return doEvaluate(y!.move(gen, κ));
  }
  
  final override function doGrad(gen:Integer) {
    y!.grad(gen, doEvaluateGrad(d!, x!, y!.get()));
  }

  final override function doPrior(vars:RaggedArray<DelayExpression>) ->
      Expression<Real>? {
    return y!.prior(vars);
  }

  final override function doConstant() {
    y!.constant();
  }

  final override function doCount(gen:Integer) {
    y!.count(gen);
  }

  final override function doDetach() {
    y <- nil;
  }
}
