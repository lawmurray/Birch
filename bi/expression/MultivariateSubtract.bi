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

  function graftMultivariateLinearGaussian() ->
      TransformLinearMultivariateGaussian? {
    y:TransformLinearMultivariateGaussian?;
    z:DelayMultivariateGaussian?;

    if (y <- left.graftMultivariateLinearGaussian())? {
      y!.subtract(right.value());
    } else if (y <- right.graftMultivariateLinearGaussian())? {
      y!.negateAndAdd(left.value());
    } else if (z <- left.graftMultivariateGaussian())? {
      y <- TransformLinearMultivariateGaussian(identity(z!.size()), z!,
          -right.value());
    } else if (z <- right.graftMultivariateGaussian())? {
      y <- TransformLinearMultivariateGaussian(diagonal(-1, z!.size()), z!,
          left.value());
    }
    return y;
  }
  
  function graftMultivariateLinearNormalInverseGamma() ->
      TransformLinearIdenticalNormalInverseGamma? {
    y:TransformLinearIdenticalNormalInverseGamma?;
    z:DelayIdenticalNormalInverseGamma?;

    if (y <- left.graftMultivariateLinearNormalInverseGamma())? {
      y!.subtract(right.value());
    } else if (y <- right.graftMultivariateLinearNormalInverseGamma())? {
      y!.negateAndAdd(left.value());
    } else if (z <- left.graftIdenticalNormalInverseGamma())? {
      y <- TransformLinearIdenticalNormalInverseGamma(identity(z!.size()),
          z!, -right.value());
    } else if (z <- right.graftIdenticalNormalInverseGamma())? {
      y <- TransformLinearIdenticalNormalInverseGamma(
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
