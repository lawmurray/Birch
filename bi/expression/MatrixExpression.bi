/**
 * Expression that evaluates to a matrix.
 *
 * - Value: Matrix type.
 */
abstract class MatrixExpression<Value> < Expression<Value>(nil) {  
  /**
   * Accumulated upstream gradient.
   */
  d:Real[_,_]?;

  final override function rows() -> Integer {
    if x? {
      return global.rows(x!);
    } else {
      return doRows();
    }
  }
  
  final override function columns() -> Integer {
    if x? {
      return global.columns(x!);
    } else {
      return doColumns();
    }
  }
  
  function doRows() -> Integer {
    return 1;
  }
  
  function doColumns() -> Integer {
    return 1;
  }

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
