/**
 * Scalar binary expression.
 *
 * - Left: Left argument type.
 * - Right: Right argument type.
 * - Value: Value type.
 */
abstract class ScalarBinaryExpression<Left,Right,Value>(left:Left,
    right:Right) < ScalarExpression<Value> {  
  /**
   * Left argument.
   */
  left:Left <- left;
  
  /**
   * Right argument.
   */
  right:Right <- right;
  
  final override function doMakeConstant() {
    left.makeConstant();
    right.makeConstant();
  }

  final override function doRestoreCount() {
    left.restoreCount();
    right.restoreCount();
  }
  
  final override function doPrior(vars:RaggedArray<DelayExpression>) ->
      Expression<Real>? {
    auto l <- left.prior(vars);
    auto r <- right.prior(vars);
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
