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
      TransformLinearMultivariate<DelayMultivariateGaussian>? {
    y:TransformLinearMultivariate<DelayMultivariateGaussian>?;
    z:DelayMultivariateGaussian?;

    if (y <- left.graftLinearMultivariateGaussian())? {
      y!.subtract(right.value());
    } else if (y <- right.graftLinearMultivariateGaussian())? {
      y!.negateAndAdd(left.value());
    } else if (z <- left.graftMultivariateGaussian())? {
      y <- TransformLinearMultivariate<DelayMultivariateGaussian>(
          identity(z!.rows()), z!, -right.value());
    } else if (z <- right.graftMultivariateGaussian())? {
      y <- TransformLinearMultivariate<DelayMultivariateGaussian>(
          diagonal(-1, z!.rows()), z!, left.value());
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
          identity(z!.rows()), z!, -right.value());
    } else if (z <- right.graftMultivariateNormalInverseGamma())? {
      y <- TransformLinearMultivariate<DelayMultivariateNormalInverseGamma>(
          diagonal(-1, z!.rows()), z!, left.value());
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
