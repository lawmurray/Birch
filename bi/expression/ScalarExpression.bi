/**
 * Expression that evaluates to a scalar.
 *
 * - Value: Scalar type.
 */
abstract class ScalarExpression<Value> < Expression<Value> {  
  /**
   * Accumulated upstream gradient.
   */
  d:Real?;  

  final function doAccumulateGrad(d:Real) {
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
