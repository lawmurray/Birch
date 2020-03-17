/**
 * Lazy multivariate subtract.
 */
final class MultivariateSubtract<Left,Right,Value>(left:Expression<Left>,
    right:Expression<Right>) < BinaryExpression<Left,Right,Value>(left, right) {  
  function rows() -> Integer {
    assert left.rows() == right.rows();
    return left.rows();
  }

  function doValue(l:Left, r:Right) -> Value {
    return l - r;
  }

  function doGradient(d:Value, l:Left, r:Right) -> (Left, Right) {
    return (d, -d);
  }

  function graftLinearMultivariateGaussian() ->
      TransformLinearMultivariate<MultivariateGaussian>? {
    y:TransformLinearMultivariate<MultivariateGaussian>?;
    z:MultivariateGaussian?;

    if (y <- left.graftLinearMultivariateGaussian())? {
      y!.subtract(right);
    } else if (y <- right.graftLinearMultivariateGaussian())? {
      y!.negateAndAdd(left);
    } else if (z <- left.graftMultivariateGaussian())? {
      y <- TransformLinearMultivariate<MultivariateGaussian>(Identity(z!.rows()), z!, -right);
    } else if (z <- right.graftMultivariateGaussian())? {
      y <- TransformLinearMultivariate<MultivariateGaussian>(-Identity(z!.rows()), z!, left);
    }
    return y;
  }
  
  function graftLinearMultivariateNormalInverseGamma(compare:Distribution<Real>) ->
      TransformLinearMultivariate<MultivariateNormalInverseGamma>? {
    y:TransformLinearMultivariate<MultivariateNormalInverseGamma>?;
    z:MultivariateNormalInverseGamma?;

    if (y <- left.graftLinearMultivariateNormalInverseGamma(compare))? {
      y!.subtract(right);
    } else if (y <- right.graftLinearMultivariateNormalInverseGamma(compare))? {
      y!.negateAndAdd(left);
    } else if (z <- left.graftMultivariateNormalInverseGamma(compare))? {
      y <- TransformLinearMultivariate<MultivariateNormalInverseGamma>(Identity(z!.rows()), z!, -right);
    } else if (z <- right.graftMultivariateNormalInverseGamma(compare))? {
      y <- TransformLinearMultivariate<MultivariateNormalInverseGamma>(-Identity(z!.rows()), z!, left);
    }
    return y;
  }
}

/**
 * Lazy multivariate subtract.
 */
operator (left:Expression<Real[_]> - right:Expression<Real[_]>) ->
    MultivariateSubtract<Real[_],Real[_],Real[_]> {
  assert left.rows() == right.rows();
  m:MultivariateSubtract<Real[_],Real[_],Real[_]>(left, right);
  return m;
}

/**
 * Lazy multivariate subtract.
 */
operator (left:Real[_] - right:Expression<Real[_]>) ->
    MultivariateSubtract<Real[_],Real[_],Real[_]> {
  return Boxed(left) - right;
}

/**
 * Lazy multivariate subtract.
 */
operator (left:Expression<Real[_]> - right:Real[_]) ->
    MultivariateSubtract<Real[_],Real[_],Real[_]> {
  return left - Boxed(right);
}
