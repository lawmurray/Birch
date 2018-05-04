/**
 * Lazy multivariate addition.
 */
class MultivariateAdd<Left,Right,Value>(left:Expression<Left>,
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
    return left.value() + right.value();
  }

  function graftMultivariateLinearGaussian() ->
      TransformMultivariateLinearGaussian? {
    y:TransformMultivariateLinearGaussian?;
    z:DelayMultivariateGaussian?;

    if (y <- left.graftMultivariateLinearGaussian())? {
      y!.add(right.value());
    } else if (y <- right.graftMultivariateLinearGaussian())? {
      y!.add(left.value());
    } else if (z <- left.graftMultivariateGaussian())? {
      y <- TransformMultivariateLinearGaussian(identity(z!.size()), z!,
          right.value());
    } else if (z <- right.graftMultivariateGaussian())? {
      y <- TransformMultivariateLinearGaussian(identity(z!.size()), z!,
          left.value());
    }
    return y;
  }
  
  function graftMultivariateLinearNormalInverseGamma(σ2:Expression<Real>) ->
      TransformMultivariateLinearNormalInverseGamma? {
    y:TransformMultivariateLinearNormalInverseGamma?;
    z:DelayMultivariateNormalInverseGamma?;

    if (y <- left.graftMultivariateLinearNormalInverseGamma(σ2))? {
      y!.add(right.value());
    } else if (y <- right.graftMultivariateLinearNormalInverseGamma(σ2))? {
      y!.add(left.value());
    } else if (z <- left.graftMultivariateNormalInverseGamma(σ2))? {
      y <- TransformMultivariateLinearNormalInverseGamma(identity(z!.size()),
          z!, right.value());
    } else if (z <- right.graftMultivariateNormalInverseGamma(σ2))? {
      y <- TransformMultivariateLinearNormalInverseGamma(identity(z!.size()),
          z!, left.value());
    }
    return y;
  }
}

operator (left:Expression<Real[_]> + right:Expression<Real[_]>) ->
    MultivariateAdd<Real[_],Real[_],Real[_]> {
  m:MultivariateAdd<Real[_],Real[_],Real[_]>(left, right);
  return m;
}

operator (left:Real[_] + right:Expression<Real[_]>) ->
    MultivariateAdd<Real[_],Real[_],Real[_]> {
  return Boxed(left) + right;
}

operator (left:Expression<Real[_]> + right:Real[_]) ->
    MultivariateAdd<Real[_],Real[_],Real[_]> {
  return left + Boxed(right);
}
