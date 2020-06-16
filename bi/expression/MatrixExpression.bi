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

  final override function element(i:Expression<Integer>,
      j:Expression<Integer>) -> Expression<Real> {
    return MatrixElement(this, i, j);
  }

  final override function doAccumulateGrad(D:Real[_,_]) {
    if this.D? {
      this.D <- this.D! + D;
    } else {
      this.D <- D;
    }
  }

  final override function doAccumulateGrad(d:Real, i:Integer, j:Integer) {
    if !this.D? {
      this.D <- matrix(0.0, rows(), columns());
    }
    this.D![i,j] <- this.D![i,j] + d;
  }
  
  final override function doClearGrad() {
    this.D <- nil;
  }
}
