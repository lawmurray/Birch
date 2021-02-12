/**
 * Expression that evaluates to a matrix.
 *
 * - Value: Matrix type.
 */
abstract class MatrixExpression<Value> < Expression<Value> {  
  /**
   * Accumulated upstream gradient.
   */
  d:Real[_,_]?;

  final override function doAccumulateGrad(d:Real[_,_]) {
    if this.d? {
      this.d <- this.d! + d;
    } else {
      this.d <- d;
    }
  }

  final override function doAccumulateGrad(d:Real, i:Integer, j:Integer) {
    if !this.d? {
      this.d <- matrix(0.0, rows(), columns());
    }
    this.d![i,j] <- this.d![i,j] + d;
  }
  
  final override function doClearGrad() {
    this.d <- nil;
  }
}
