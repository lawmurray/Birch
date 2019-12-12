/**
 * Lazy multiply.
 */
final class Multiply<Left,Right,Value>(left:Expression<Left>,
    right:Expression<Right>) < BinaryExpression<Left,Right,Value>(left, right) {  
  function graft() -> Expression<Value> {
    return left.graft()*right.graft();
  }

  function doValue(l:Left, r:Right) -> Value {
    return l*r;
  }

  function doGradient(d:Value, l:Left, r:Right) -> (Left, Right) {
    return (d*r, d*l);
  }

  function graftScaledGamma() -> TransformLinear<DelayGamma>? {
    y:TransformLinear<DelayGamma>?;
    z:DelayGamma?;
    
    if (y <- left.graftScaledGamma())? {
      y!.multiply(right);
    } else if (y <- right.graftScaledGamma())? {
      y!.multiply(left);
    } else if (z <- left.graftGamma())? {
      y <- TransformLinear<DelayGamma>(right, z!);
    } else if (z <- right.graftGamma())? {
      y <- TransformLinear<DelayGamma>(left, z!);
    }
    return y;
  }

  function graftLinearGaussian() -> TransformLinear<DelayGaussian>? {
    y:TransformLinear<DelayGaussian>?;
    z:DelayGaussian?;
    
    if (y <- left.graftLinearGaussian())? {
      y!.multiply(right);
    } else if (y <- right.graftLinearGaussian())? {
      y!.multiply(left);
    } else if (z <- left.graftGaussian())? {
      y <- TransformLinear<DelayGaussian>(right, z!);
    } else if (z <- right.graftGaussian())? {
      y <- TransformLinear<DelayGaussian>(left, z!);
    }
    return y;
  }

  function graftDotGaussian() -> TransformDot<DelayMultivariateGaussian>? {
    y:TransformDot<DelayMultivariateGaussian>?;
    
    if (y <- left.graftDotGaussian())? {
      y!.multiply(right);
    } else if (y <- right.graftDotGaussian())? {
      y!.multiply(left);
    }
    return y;
  }
 
  function graftLinearNormalInverseGamma() ->
      TransformLinear<DelayNormalInverseGamma>? {
    y:TransformLinear<DelayNormalInverseGamma>?;
    z:DelayNormalInverseGamma?;
    
    if (y <- left.graftLinearNormalInverseGamma())? {
      y!.multiply(right);
    } else if (y <- right.graftLinearNormalInverseGamma())? {
      y!.multiply(left);
    } else if (z <- left.graftNormalInverseGamma())? {
      y <- TransformLinear<DelayNormalInverseGamma>(right, z!);
    } else if (z <- right.graftNormalInverseGamma())? {
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
