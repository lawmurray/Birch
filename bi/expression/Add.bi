/**
 * Lazy add.
 */
final class Add<Left,Right,Value>(left:Expression<Left>,
    right:Expression<Right>) < BinaryExpression<Left,Right,Value>(left, right) {  
  function doValue(l:Left, r:Right) -> Value {
    return l + r;
  }
  
  function doGradient(d:Value, l:Left, r:Right) -> (Left, Right) {
    return (d, d);
  }

  function graftLinearGaussian() -> TransformLinear<DelayGaussian>? {
    y:TransformLinear<DelayGaussian>?;
    z:DelayGaussian?;
    
    if (y <- left.graftLinearGaussian())? {
      y!.add(right);
    } else if (y <- right.graftLinearGaussian())? {
      y!.add(left);
    } else if (z <- left.graftGaussian())? {
      y <- TransformLinear<DelayGaussian>(Boxed(1.0), z!, right);
    } else if (z <- right.graftGaussian())? {
      y <- TransformLinear<DelayGaussian>(Boxed(1.0), z!, left);
    }
    return y;
  }

  function graftDotGaussian() -> TransformDot<DelayMultivariateGaussian>? {
    y:TransformDot<DelayMultivariateGaussian>?;
    z:DelayGaussian?;
    if (y <- left.graftDotGaussian())? {
      y!.add(right);
    } else if (y <- right.graftDotGaussian())? {
      y!.add(left);
    }
    return y;
  }
  
  function graftLinearNormalInverseGamma() ->
      TransformLinear<DelayNormalInverseGamma>? {
    y:TransformLinear<DelayNormalInverseGamma>?;
    z:DelayNormalInverseGamma?;

    if (y <- left.graftLinearNormalInverseGamma())? {
      y!.add(right);
    } else if (y <- right.graftLinearNormalInverseGamma())? {
      y!.add(left);
    } else if (z <- left.graftNormalInverseGamma())? {
      y <- TransformLinear<DelayNormalInverseGamma>(Boxed(1.0), z!, right);
    } else if (z <- right.graftNormalInverseGamma())? {
      y <- TransformLinear<DelayNormalInverseGamma>(Boxed(1.0), z!, left);
    }
    return y;
  }
}

/**
 * Lazy add.
 */
operator (left:Expression<Real> + right:Expression<Real>) ->
    Add<Real,Real,Real> {
  m:Add<Real,Real,Real>(left, right);
  return m;
}

/**
 * Lazy add.
 */
operator (left:Real + right:Expression<Real>) -> Add<Real,Real,Real> {
  return Boxed(left) + right;
}

/**
 * Lazy add.
 */
operator (left:Expression<Real> + right:Real) -> Add<Real,Real,Real> {
  return left + Boxed(right);
}
