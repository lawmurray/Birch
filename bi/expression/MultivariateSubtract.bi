/*
 * Lazy multivariate subtraction.
 */
final class MultivariateSubtract<Left,Right,Value>(left:Expression<Left>,
    right:Expression<Right>) < Expression<Value> {  
  /**
   * Left operand.
   */
  left:Expression<Left> <- left;
  
  /**
   * Right operand.
   */
  right:Expression<Right> <- right;

  function value() -> Value {
    return left.value() - right.value();
  }

  function graftLinearMultivariateGaussian() ->
      TransformLinearMultivariate<DelayMultivariateGaussian>? {
    y:TransformLinearMultivariate<DelayMultivariateGaussian>?;
    z:DelayMultivariateGaussian?;

    if (y <- left.graftLinearMultivariateGaussian())? {
      y!.subtract(right.value());
    } else if (y <- right.graftLinearMultivariateGaussian())? {
      y!.negateAndAdd(left.value());
    } else if (z <- left.graftMultivariateGaussian())? {
      y <- TransformLinearMultivariate<DelayMultivariateGaussian>(
          identity(z!.size()), z!, -right.value());
    } else if (z <- right.graftMultivariateGaussian())? {
      y <- TransformLinearMultivariate<DelayMultivariateGaussian>(
          diagonal(-1, z!.size()), z!, left.value());
    }
    return y;
  }
  
  function graftLinearMultivariateNormalInverseGamma() ->
      TransformLinearMultivariate<DelayMultivariateNormalInverseGamma>? {
    y:TransformLinearMultivariate<DelayMultivariateNormalInverseGamma>?;
    z:DelayMultivariateNormalInverseGamma?;

    if (y <- left.graftLinearMultivariateNormalInverseGamma())? {
      y!.subtract(right.value());
    } else if (y <- right.graftLinearMultivariateNormalInverseGamma())? {
      y!.negateAndAdd(left.value());
    } else if (z <- left.graftMultivariateNormalInverseGamma())? {
      y <- TransformLinearMultivariate<DelayMultivariateNormalInverseGamma>(
          identity(z!.size()), z!, -right.value());
    } else if (z <- right.graftMultivariateNormalInverseGamma())? {
      y <- TransformLinearMultivariate<DelayMultivariateNormalInverseGamma>(
          diagonal(-1, z!.size()), z!, left.value());
    }
    return y;
  }
}

operator (left:Expression<Real[_]> - right:Expression<Real[_]>) ->
    MultivariateSubtract<Real[_],Real[_],Real[_]> {
  m:MultivariateSubtract<Real[_],Real[_],Real[_]>(left, right);
  return m;
}

operator (left:Real[_] - right:Expression<Real[_]>) ->
    MultivariateSubtract<Real[_],Real[_],Real[_]> {
  return Boxed(left) - right;
}

operator (left:Expression<Real[_]> - right:Real[_]) ->
    MultivariateSubtract<Real[_],Real[_],Real[_]> {
  return left - Boxed(right);
}
