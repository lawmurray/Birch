/**
 * Lazy subtract.
 */
final class Subtract<Left,Right,Value>(left:Expression<Left>,
    right:Expression<Right>) < BinaryExpression<Left,Right,Value>(left, right) {  
  function graft(child:Delay?) -> Expression<Value> {
    return left.graft(child) - right.graft(child);
  }

  function doValue(l:Left, r:Right) -> Value {
    return l - r;
  }
  
  function doGradient(d:Value, l:Left, r:Right) -> (Left, Right) {
    return (d, -d);
  }

  function graftLinearGaussian(child:Delay?) -> TransformLinear<DelayGaussian>? {
    y:TransformLinear<DelayGaussian>?;
    z:DelayGaussian?;
    
    if (y <- left.graftLinearGaussian(child))? {
      y!.add(-right);
    } else if (y <- right.graftLinearGaussian(child))? {
      y!.negateAndAdd(left);
    } else if (z <- left.graftGaussian(child))? {
      y <- TransformLinear<DelayGaussian>(Boxed(1.0), z!, -right);
    } else if (z <- right.graftGaussian(child))? {
      y <- TransformLinear<DelayGaussian>(Boxed(-1.0), z!, left);
    }
    return y;
  }

  function graftDotGaussian(child:Delay?) -> TransformDot<DelayMultivariateGaussian>? {
    y:TransformDot<DelayMultivariateGaussian>?;
    z:DelayGaussian?;
    
    if (y <- left.graftDotGaussian(child))? {
      y!.add(-right);
    } else if (y <- right.graftDotGaussian(child))? {
      y!.negateAndAdd(left);
    }
    return y;
  }

  function graftLinearNormalInverseGamma(child:Delay?) ->
      TransformLinear<DelayNormalInverseGamma>? {
    y:TransformLinear<DelayNormalInverseGamma>?;
    z:DelayNormalInverseGamma?;

    if (y <- left.graftLinearNormalInverseGamma(child))? {
      y!.subtract(right);
    } else if (y <- right.graftLinearNormalInverseGamma(child))? {
      y!.negateAndAdd(left);
    } else if (z <- left.graftNormalInverseGamma(child))? {
      y <- TransformLinear<DelayNormalInverseGamma>(Boxed(1.0), z!, -right);
    } else if (z <- right.graftNormalInverseGamma(child))? {
      y <- TransformLinear<DelayNormalInverseGamma>(Boxed(-1.0), z!, left);
    }
    return y;
  }
}

/**
 * Lazy subtract.
 */
operator (left:Expression<Real> - right:Expression<Real>) ->
    Subtract<Real,Real,Real> {
  m:Subtract<Real,Real,Real>(left, right);
  return m;
}

/**
 * Lazy subtract.
 */
operator (left:Real - right:Expression<Real>) -> Subtract<Real,Real,Real> {
  return Boxed(left) - right;
}

/**
 * Lazy subtract.
 */
operator (left:Expression<Real> - right:Real) -> Subtract<Real,Real,Real> {
  return left - Boxed(right);
}
