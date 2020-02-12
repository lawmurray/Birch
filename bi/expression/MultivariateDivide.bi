/**
 * Lazy multivariate divide (by scalar).
 */
final class MultivariateDivide<Left,Right,Value>(left:Expression<Left>,
    right:Expression<Right>) < BinaryExpression<Left,Right,Value>(left, right) {  
  function rows() -> Integer {
    return left.rows();
  }

  function doValue(l:Left, r:Right) -> Value {
    return l/r;
  }

  function doGradient(d:Value, l:Left, r:Right) -> (Left, Right) {
    return (d/r, transpose(l)*(-d/(r*r)));
  }

  function graftLinearMultivariateGaussian() ->
      TransformLinearMultivariate<MultivariateGaussian>? {
    y:TransformLinearMultivariate<MultivariateGaussian>?;
    z:MultivariateGaussian?;
    
    if (y <- left.graftLinearMultivariateGaussian())? {
      y!.divide(right);
    } else if (z <- left.graftMultivariateGaussian())? {
      y <- TransformLinearMultivariate<MultivariateGaussian>(
          diagonal(1.0/right, z!.rows()), z!);
    }
    return y;
  }
  
  function graftLinearMultivariateNormalInverseGamma() ->
      TransformLinearMultivariate<MultivariateNormalInverseGamma>? {
    y:TransformLinearMultivariate<MultivariateNormalInverseGamma>?;
    z:MultivariateNormalInverseGamma?;

    if (y <- left.graftLinearMultivariateNormalInverseGamma())? {
      y!.divide(right);
    } else if (z <- left.graftMultivariateNormalInverseGamma())? {
      y <- TransformLinearMultivariate<MultivariateNormalInverseGamma>(
          diagonal(1.0/right, z!.rows()), z!);
    }
    return y;
  }
}

/**
 * Lazy multivariate divide.
 */
operator (left:Expression<Real[_]>/right:Expression<Real>) ->
    MultivariateDivide<Real[_],Real,Real[_]> {
  m:MultivariateDivide<Real[_],Real,Real[_]>(left, right);
  return m;
}

/**
 * Lazy multivariate divide.
 */
operator (left:Real[_]/right:Expression<Real>) ->
    MultivariateDivide<Real[_],Real,Real[_]> {
  return Boxed(left)/right;
}

/**
 * Lazy multivariate divide.
 */
operator (left:Expression<Real[_]>/right:Real) ->
    MultivariateDivide<Real[_],Real,Real[_]> {
  return left/Boxed(right);
}
