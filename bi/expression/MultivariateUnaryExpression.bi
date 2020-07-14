/**
 * Multivariate unary expression.
 *
 * - `Argument`: Argument type. Should derive from `Expression<...>`.
 * - `ArgumentValue`: Argument value type. This is the type to which the left
 *   argument evaluates.
 * - `ArgumentGradient`: Argument upstream gradient type. This is the type of
 *   the upstream gradient that the argument accepts. It should be `Real`,
 *   `Real[_]`, or `Real[_,_]`.
 * - `Value`: The type to which the expression evaluates.
 */
abstract class MultivariateUnaryExpression<Argument,ArgumentValue,
    ArgumentGradient,Value>(y:Argument) < MultivariateExpression<Value> {  
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
  abstract function doEvaluateGrad(d:Real[_], x:Value, y:ArgumentValue) ->
      ArgumentGradient;

  final override function doDepth() -> Integer {
    return y!.depth() + 1;
  }

  final override function doValue() -> Value {
    return doEvaluate(y!.value());
  }

  final override function doPilot() -> Value {
    return doEvaluate(y!.pilot());
  }

  final override function doGet() -> Value {
    return doEvaluate(y!.get());
  }

  final override function doMove(κ:Kernel) -> Value {
    return doEvaluate(y!.move(κ));
  }
  
  final override function doGrad() {
    y!.grad(doEvaluateGrad(d!, x!, y!.get()));
  }

  final override function doPrior(vars:RaggedArray<DelayExpression>) ->
      Expression<Real>? {
    return y!.prior(vars);
  }

  final override function doConstant() {
    y!.constant();
  }

  final override function doCount() {
    y!.count();
  }

  final override function doDetach() {
    y <- nil;
  }
}
