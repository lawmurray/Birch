/**
 * Lazy multivariate subtraction.
 */
class MultivariateSubtract<Left,Right,Value>(left:Expression<Left>,
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
      TransformMultivariateLinearGaussian? {
    y:TransformMultivariateLinearGaussian?;
    z:DelayMultivariateGaussian?;

    if (y <- left.graftMultivariateLinearGaussian())? {
      y!.subtract(right.value());
    } else if (y <- right.graftMultivariateLinearGaussian())? {
      y!.negateAndAdd(left.value());
    } else if (z <- left.graftMultivariateGaussian())? {
      y <- TransformMultivariateLinearGaussian(identity(z!.size()), z!,
          -right.value());
    } else if (z <- right.graftMultivariateGaussian())? {
      y <- TransformMultivariateLinearGaussian(diagonal(-1, z!.size()), z!,
          left.value());
    }
    return y;
  }
  
  function graftMultivariateLinearNormalInverseGamma(σ2:Expression<Real>) ->
      TransformMultivariateLinearNormalInverseGamma? {
    y:TransformMultivariateLinearNormalInverseGamma?;
    z:DelayMultivariateNormalInverseGamma?;

    if (y <- left.graftMultivariateLinearNormalInverseGamma(σ2))? {
      y!.subtract(right.value());
    } else if (y <- right.graftMultivariateLinearNormalInverseGamma(σ2))? {
      y!.negateAndAdd(left.value());
    } else if (z <- left.graftMultivariateNormalInverseGamma(σ2))? {
      y <- TransformMultivariateLinearNormalInverseGamma(identity(z!.size()),
          z!, -right.value());
    } else if (z <- right.graftMultivariateNormalInverseGamma(σ2))? {
      y <- TransformMultivariateLinearNormalInverseGamma(
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
