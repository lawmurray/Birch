/**
 * Matrix unary expression.
 *
 * - Single: Single argument type.
 * - Value: Value type.
 */
abstract class MatrixUnaryExpression<Single,Value>(single:Single) <
    MatrixExpression<Value> {  
  /**
   * Single argument.
   */
  single:Single? <- single;

  final override function doDetach() {
    single <- nil;
  }

  final override function doMakeConstant() {
    single!.makeConstant();
  }

  final override function doPrior(vars:RaggedArray<DelayExpression>) ->
      Expression<Real>? {
    return single!.prior(vars);
  }
}
