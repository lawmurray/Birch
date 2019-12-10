/**
 * Lazy multivariate multiply.
 */
final class MultivariateMultiply<Left,Right,Value>(left:Expression<Left>,
    right:Expression<Right>) < BinaryExpression<Left,Right,Value>(left, right) {  
  function rows() -> Integer {
    return left.rows();
  }
  
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
      y!.leftMultiply(left);
    } else if (z <- right.graftMultivariateGaussian())? {
      y <- TransformLinearMultivariate<DelayMultivariateGaussian>(
          left, z!);
    }
    return y;
  }
  
  function graftLinearMultivariateNormalInverseGamma() ->
      TransformLinearMultivariate<DelayMultivariateNormalInverseGamma>? {
    y:TransformLinearMultivariate<DelayMultivariateNormalInverseGamma>?;
    z:DelayMultivariateNormalInverseGamma?;

    if (y <- right.graftLinearMultivariateNormalInverseGamma())? {
      y!.leftMultiply(left);
    } else if (z <- right.graftMultivariateNormalInverseGamma())? {
      y <- TransformLinearMultivariate<DelayMultivariateNormalInverseGamma>(
          left, z!);
    }
    return y;
  }
}

/**
 * Lazy multivariate multiply.
 */
operator (left:Expression<Real[_,_]>*right:Expression<Real[_]>) ->
    MultivariateMultiply<Real[_,_],Real[_],Real[_]> {
  assert left.columns() == right.rows();
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

/**
 * Lazy multivariate multiply.
 */
operator (left:Expression<Real>*right:Expression<Real[_]>) ->
    MultivariateMultiply<Real[_,_],Real[_],Real[_]> {
  m:MultivariateMultiply<Real[_,_],Real[_],Real[_]>(diagonal(left,
      right.rows()), right);
  return m;
}

/**
 * Lazy multivariate multiply.
 */
operator (left:Real*right:Expression<Real[_]>) ->
    MultivariateMultiply<Real[_,_],Real[_],Real[_]> {
  return diagonal(Boxed(left), right.rows())*right;
}

/**
 * Lazy multivariate multiply.
 */
operator (left:Expression<Real>*right:Real[_]) ->
    MultivariateMultiply<Real[_,_],Real[_],Real[_]> {
  return diagonal(left, rows(right))*Boxed(right);
}

/**
 * Lazy multivariate multiply.
 */
operator (left:Expression<Real[_]>*right:Expression<Real>) ->
    MultivariateMultiply<Real[_,_],Real[_],Real[_]> {
  m:MultivariateMultiply<Real[_,_],Real[_],Real[_]>(diagonal(right,
      left.rows()), left);
  return m;
}

/**
 * Lazy multivariate multiply.
 */
operator (left:Real[_]*right:Expression<Real>) ->
    MultivariateMultiply<Real[_,_],Real[_],Real[_]> {
  return diagonal(right, right.rows())*Boxed(left);
}

/**
 * Lazy multivariate multiply.
 */
operator (left:Expression<Real[_]>*right:Real) ->
    MultivariateMultiply<Real[_,_],Real[_],Real[_]> {
  return diagonal(Boxed(right), left.rows())*left;
}
