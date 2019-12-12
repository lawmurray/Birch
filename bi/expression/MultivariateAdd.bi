/**
 * Lazy multivariate add.
 */
final class MultivariateAdd<Left,Right,Value>(left:Expression<Left>,
    right:Expression<Right>) < BinaryExpression<Left,Right,Value>(left, right) {  
  function rows() -> Integer {
    assert left.rows() == right.rows();
    return left.rows();
  }

  function graft(child:Delay?) -> Expression<Value> {
    return left.graft(child) + right.graft(child);
  }


  function doValue(l:Left, r:Right) -> Value {
    return l + r;
  }

  function doGradient(d:Value, l:Left, r:Right) -> (Left, Right) {
    return (d, d);
  }

  function graftLinearMultivariateGaussian(child:Delay?) ->
      TransformLinearMultivariate<DelayMultivariateGaussian>? {
    y:TransformLinearMultivariate<DelayMultivariateGaussian>?;
    z:DelayMultivariateGaussian?;

    if (y <- left.graftLinearMultivariateGaussian(child))? {
      y!.add(right);
    } else if (y <- right.graftLinearMultivariateGaussian(child))? {
      y!.add(left);
    } else if (z <- left.graftMultivariateGaussian(child))? {
      y <- TransformLinearMultivariate<DelayMultivariateGaussian>(
          Boxed(identity(z!.rows())), z!, right);
    } else if (z <- right.graftMultivariateGaussian(child))? {
      y <- TransformLinearMultivariate<DelayMultivariateGaussian>(
          Boxed(identity(z!.rows())), z!, left);
    }
    return y;
  }
  
  function graftLinearMultivariateNormalInverseGamma(child:Delay?) ->
      TransformLinearMultivariate<DelayMultivariateNormalInverseGamma>? {
    y:TransformLinearMultivariate<DelayMultivariateNormalInverseGamma>?;
    z:DelayMultivariateNormalInverseGamma?;

    if (y <- left.graftLinearMultivariateNormalInverseGamma(child))? {
      y!.add(right);
    } else if (y <- right.graftLinearMultivariateNormalInverseGamma(child))? {
      y!.add(left);
    } else if (z <- left.graftMultivariateNormalInverseGamma(child))? {
      y <- TransformLinearMultivariate<DelayMultivariateNormalInverseGamma>(
          Boxed(identity(z!.rows())), z!, right);
    } else if (z <- right.graftMultivariateNormalInverseGamma(child))? {
      y <- TransformLinearMultivariate<DelayMultivariateNormalInverseGamma>(
          Boxed(identity(z!.rows())), z!, left);
    }
    return y;
  }
}

/**
 * Lazy multivariate add.
 */
operator (left:Expression<Real[_]> + right:Expression<Real[_]>) ->
    MultivariateAdd<Real[_],Real[_],Real[_]> {
  assert left.rows() == right.rows();
  m:MultivariateAdd<Real[_],Real[_],Real[_]>(left, right);
  return m;
}

/**
 * Lazy multivariate add.
 */
operator (left:Real[_] + right:Expression<Real[_]>) ->
    MultivariateAdd<Real[_],Real[_],Real[_]> {
  return Boxed(left) + right;
}

/**
 * Lazy multivariate add.
 */
operator (left:Expression<Real[_]> + right:Real[_]) ->
    MultivariateAdd<Real[_],Real[_],Real[_]> {
  return left + Boxed(right);
}
