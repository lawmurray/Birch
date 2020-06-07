/**
 * Expression that evaluates to a matrix.
 *
 * - Value: Matrix type.
 */
abstract class MatrixExpression<Value> < Expression<Value> {  
  /**
   * Accumulated upstream gradient.
   */
  D:Real[_,_]?;

  final function doAccumulateGrad(D:Real[_,_]) {
    if this.D? {
      this.D <- this.D! + D;
    } else {
      this.D <- D;
    }
  }
  
  final function doClearGrad() {
    D <- nil;
  }
}
