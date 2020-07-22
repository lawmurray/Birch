/**
 * Scalar binary expression.
 *
 * - `Left`: Left argument type. Should derive from `Expression<...>`.
 * - `Right`: Right argument type. Should derive from `Expression<...>`.
 * - `LeftValue`: Left value type. This is the type to which the left
 *    argument evaluates.
 * - `RightValue`: Right value type. This is the type to which the right
 *   argument evaluates.
 * - `LeftGradient`: Left upstream gradient type. This is the type of the
 *   upstream gradient that the left argument accepts. It should be `Real`,
 *   `Real[_]`, or `Real[_,_]`.
 * - `RightGradient`: Right upstrem gradient type. This is the type of the
 *   upstream gradient that the right argument accepts. It should be `Real`,
 *   `Real[_]`, or `Real[_,_]`.
 * - `Value`: The type to which the expression evaluates.
 */
abstract class ScalarBinaryExpression<Left,Right,LeftValue,RightValue,
    LeftGradient,RightGradient,Value>(y:Left, z:Right) <
    ScalarExpression<Value> {  
  /**
   * Left argument.
   */
  y:Left? <- y;
  
  /**
   * Right argument.
   */
  z:Right? <- z;

  /*
   * Evaluate.
   */
  abstract function doEvaluate(y:LeftValue, z:RightValue) -> Value;
  
  /*
   * Evaluate the gradient for the left argument.
   */
  abstract function doEvaluateGradLeft(d:Real, x:Value, y:LeftValue,
      z:RightValue) -> LeftGradient;

  /*
   * Evaluate the gradient for the right argument.
   */
  abstract function doEvaluateGradRight(d:Real, x:Value, y:LeftValue,
      z:RightValue) -> RightGradient;


  final override function doDepth() -> Integer {
    return max(y!.depth(), z!.depth()) + 1;
  }

  final override function doValue() -> Value {
    return doEvaluate(y!.value(), z!.value());
  }

  final override function doPilot(gen:Integer) -> Value {
    return doEvaluate(y!.pilot(gen), z!.pilot(gen));
  }

  final override function doGet() -> Value {
    return doEvaluate(y!.get(), z!.get());
  }

  final override function doMove(gen:Integer, κ:Kernel) -> Value {
    return doEvaluate(y!.move(gen, κ), z!.move(gen, κ));
  }
  
  final override function doGrad(gen:Integer) {
    y!.grad(gen, doEvaluateGradLeft(d!, x!, y!.get(), z!.get()));
    z!.grad(gen, doEvaluateGradRight(d!, x!, y!.get(), z!.get()));
  }
  
  final override function doPrior() -> Expression<Real>? {
    auto l <- y!.prior();
    auto r <- z!.prior();
    if l? && r? {
      return l! + r!;
    } else if l? {
      return l!;
    } else if r? {
      return r!;
    } else {
      return nil;
    }
  }

  final override function doCompare(gen:Integer, x:DelayExpression,
      κ:Kernel) -> Real {
    auto o <- ScalarBinaryExpression<Left,Right,LeftValue,RightValue,
        LeftGradient,RightGradient,Value>?(x)!;
    return y!.compare(gen, o.y!, κ) + z!.compare(gen, o.z!, κ);
  }

  final override function doConstant() {
    y!.constant();
    z!.constant();
  }

  final override function doCount(gen:Integer) {
    y!.count(gen);
    z!.count(gen);
  }
  
  final override function doDetach() {
    y <- nil;
    z <- nil;
  }
}
