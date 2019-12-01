/**
 * Lazy multivariate multiply.
 */
final class MultivariateMultiply<Left,Right,Value>(left:Expression<Left>,
    right:Expression<Right>) < BinaryExpression<Left,Right,Value>(left, right) {  
  function doValue(l:Left, r:Right) -> Value {
    return l*r;
  }

  function doGradient(d:Value, l:Left, r:Right) -> (Left, Right) {
    return (d*transpose(r), transpose(l)*d);
  }

  function graftLinearMultivariateGaussian() ->
      TransformLinearMultivariate<DelayMultivariateGaussian>? {
    y:TransformLinearMultivariate<DelayMultivariateGaussian>?;
    z:DelayMultivariateGaussian?;
    
    if (y <- right.graftLinearMultivariateGaussian())? {
      y!.leftMultiply(left.value());
    } else if (z <- right.graftMultivariateGaussian())? {
      y <- TransformLinearMultivariate<DelayMultivariateGaussian>(
          left.value(), z!);
    }
    return y;
  }
  
  function graftLinearMultivariateNormalInverseGamma() ->
      TransformLinearMultivariate<DelayMultivariateNormalInverseGamma>? {
    y:TransformLinearMultivariate<DelayMultivariateNormalInverseGamma>?;
    z:DelayMultivariateNormalInverseGamma?;

    if (y <- right.graftLinearMultivariateNormalInverseGamma())? {
      y!.leftMultiply(left.value());
    } else if (z <- right.graftMultivariateNormalInverseGamma())? {
      y <- TransformLinearMultivariate<DelayMultivariateNormalInverseGamma>(
          left.value(), z!);
    }
    return y;
  }
}

/**
 * Lazy multivariate multiply.
 */
operator (left:Expression<Real[_,_]>*right:Expression<Real[_]>) ->
    MultivariateMultiply<Real[_,_],Real[_],Real[_]> {
  m:MultivariateMultiply<Real[_,_],Real[_],Real[_]>(left, right);
  return m;
}

/**
 * Lazy multivariate multiply.
 */
operator (left:Real[_,_]*right:Expression<Real[_]>) ->
    MultivariateMultiply<Real[_,_],Real[_],Real[_]> {
  return Boxed(left)*right;
}

/**
 * Lazy multivariate multiply.
 */
operator (left:Expression<Real[_,_]>*right:Real[_]) ->
    MultivariateMultiply<Real[_,_],Real[_],Real[_]> {
  return left*Boxed(right);
}
