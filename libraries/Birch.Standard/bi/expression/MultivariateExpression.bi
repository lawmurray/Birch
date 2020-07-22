/**
 * Expression that evaluates to a vector.
 *
 * - Value: Vector type.
 */
abstract class MultivariateExpression<Value> < Expression<Value>(nil) {  
  /**
   * Accumulated upstream gradient.
   */
  d:Real[_]?;  

  final override function doColumns() -> Integer {
    return 1;
  }

  final override function doAccumulateGrad(d:Real[_]) {
    if this.d? {
      this.d <- this.d! + d;
    } else {
      this.d <- d;
    }
  }

  final override function doAccumulateGrad(d:Real, i:Integer) {
    if !this.d? {
      this.d <- vector(0.0, length());
    }
    this.d![i] <- this.d![i] + d;
  }

  final override function doClearGrad() {
    this.d <- nil;
  }
}
