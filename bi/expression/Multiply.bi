/**
 * Lazy multiply.
 */
final class Multiply<Left,Right,Value>(left:Expression<Left>,
    right:Expression<Right>) < BinaryExpression<Left,Right,Value>(left, right) {  
  function graft(child:Delay?) -> Expression<Value> {
    return left.graft(child)*right.graft(child);
  }

  function doValue(l:Left, r:Right) -> Value {
    return l*r;
  }

  function doGradient(d:Value, l:Left, r:Right) -> (Left, Right) {
    return (d*r, d*l);
  }

  function graftScaledGamma(child:Delay?) -> TransformLinear<DelayGamma>? {
    y:TransformLinear<DelayGamma>?;
    z:DelayGamma?;
    
    if (y <- left.graftScaledGamma(child))? {
      y!.multiply(right);
    } else if (y <- right.graftScaledGamma(child))? {
      y!.multiply(left);
    } else if (z <- left.graftGamma(child))? {
      y <- TransformLinear<DelayGamma>(right, z!);
    } else if (z <- right.graftGamma(child))? {
      y <- TransformLinear<DelayGamma>(left, z!);
    }
    return y;
  }

  function graftLinearGaussian(child:Delay?) -> TransformLinear<DelayGaussian>? {
    y:TransformLinear<DelayGaussian>?;
    z:DelayGaussian?;
    
    if (y <- left.graftLinearGaussian(child))? {
      y!.multiply(right);
    } else if (y <- right.graftLinearGaussian(child))? {
      y!.multiply(left);
    } else if (z <- left.graftGaussian(child))? {
      y <- TransformLinear<DelayGaussian>(right, z!);
    } else if (z <- right.graftGaussian(child))? {
      y <- TransformLinear<DelayGaussian>(left, z!);
    }
    return y;
  }

  function graftDotGaussian(child:Delay?) -> TransformDot<DelayMultivariateGaussian>? {
    y:TransformDot<DelayMultivariateGaussian>?;
    
    if (y <- left.graftDotGaussian(child))? {
      y!.multiply(right);
    } else if (y <- right.graftDotGaussian(child))? {
      y!.multiply(left);
    }
    return y;
  }
 
  function graftLinearNormalInverseGamma(child:Delay?) ->
      TransformLinear<DelayNormalInverseGamma>? {
    y:TransformLinear<DelayNormalInverseGamma>?;
    z:DelayNormalInverseGamma?;
    
    if (y <- left.graftLinearNormalInverseGamma(child))? {
      y!.multiply(right);
    } else if (y <- right.graftLinearNormalInverseGamma(child))? {
      y!.multiply(left);
    } else if (z <- left.graftNormalInverseGamma(child))? {
      y <- TransformLinear<DelayNormalInverseGamma>(right, z!);
    } else if (z <- right.graftNormalInverseGamma(child))? {
      y <- TransformLinear<DelayNormalInverseGamma>(left, z!);
    }
    return y;
  }
}

/**
 * Lazy multiply.
 */
operator (left:Expression<Real>*right:Expression<Real>) ->
    Multiply<Real,Real,Real> {
  m:Multiply<Real,Real,Real>(left, right);
  return m;
}

/**
 * Lazy multiply.
 */
operator (left:Real*right:Expression<Real>) -> Multiply<Real,Real,Real> {
  return Boxed(left)*right;
}

/**
 * Lazy multiply.
 */
operator (left:Expression<Real>*right:Real) -> Multiply<Real,Real,Real> {
  return left*Boxed(right);
}
