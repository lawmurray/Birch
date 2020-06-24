/**
 * Matrix binary expression.
 *
 * - Left: Left argument type.
 * - Right: Right argument type.
 * - Value: Value type.
 */
abstract class MatrixBinaryExpression<Left,Right,Value>(left:Left,
    right:Right) < MatrixExpression<Value> {  
  /**
   * Left argument.
   */
  left:Left? <- left;
  
  /**
   * Right argument.
   */
  right:Right? <- right;

  final override function doDetach() {
    left <- nil;
    right <- nil;
  }

  final override function doMakeConstant() {
    left!.makeConstant();
    right!.makeConstant();
  }
  
  final override function doPrior(vars:RaggedArray<DelayExpression>) ->
      Expression<Real>? {
    auto l <- left!.prior(vars);
    auto r <- right!.prior(vars);
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
}
