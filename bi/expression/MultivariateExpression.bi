/**
 * Expression that evaluates to a vector.
 *
 * - Value: Vector type.
 */
abstract class MultivariateExpression<Value> < Expression<Value> {  
  /**
   * Accumulated upstream gradient.
   */
  d:Real[_]?;  

  final function doAccumulateGrad(d:Real[_]) {
    if this.d? {
      this.d <- this.d! + d;
    } else {
      this.d <- d;
    }
  }

  final function doClearGrad() {
    d <- nil;
  }
}
