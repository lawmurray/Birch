/**
 * Delayed multivariation addition.
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

  function graftMultivariateAffineGaussian() ->
      TransformMultivariateAffineGaussian? {
    y:TransformMultivariateAffineGaussian?;
    z:DelayMultivariateGaussian?;

    if (y <- left.graftMultivariateAffineGaussian())? {
      y!.add(right.value());
    } else if (y <- right.graftMultivariateAffineGaussian())? {
      y!.add(left.value());
    } else if (z <- left.graftMultivariateGaussian())? {
      y <- TransformMultivariateAffineGaussian(identity(z!.size()), z!,
          right.value());
    } else if (z <- right.graftMultivariateGaussian())? {
      y <- TransformMultivariateAffineGaussian(identity(z!.size()), z!,
          left.value());
    }
    return y;
  }
  
  function getMultivariateAffineNormalInverseGamma(σ2:Expression<Real>) ->
      TransformMultivariateAffineNormalInverseGamma? {
    y:TransformMultivariateAffineNormalInverseGamma?;
    z:DelayMultivariateNormalInverseGamma?;

    if (y <- left.graftMultivariateAffineNormalInverseGamma(σ2))? {
      y!.add(right.value());
    } else if (y <- right.graftMultivariateAffineNormalInverseGamma(σ2))? {
      y!.add(left.value());
    } else if (z <- left.graftMultivariateNormalInverseGamma(σ2))? {
      y <- TransformMultivariateAffineNormalInverseGamma(identity(z!.size()),
          z!, right.value());
    } else if (z <- right.graftMultivariateNormalInverseGamma(σ2))? {
      y <- TransformMultivariateAffineNormalInverseGamma(identity(z!.size()),
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
