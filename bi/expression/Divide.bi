/**
 * Lazy divide.
 */
final class Divide<Left,Right,Value>(left:Expression<Left>,
    right:Expression<Right>) < BinaryExpression<Left,Right,Value>(left, right) {  
  function graft() -> Expression<Value> {
    return left.graft()/right.graft();
  }

  function doValue(l:Left, r:Right) -> Value {
    return l/r;
  }

  function doGradient(d:Value, l:Left, r:Right) -> (Left, Right) {
    return (d/r, -d*l/(r*r));
  }

  function graftLinearGaussian() -> TransformLinear<DelayGaussian>? {
    y:TransformLinear<DelayGaussian>?;
    z:DelayGaussian?;
    
    if (y <- left.graftLinearGaussian())? {
      y!.divide(right);
    } else if (z <- left.graftGaussian())? {
      y <- TransformLinear<DelayGaussian>(1.0/right, z!);
    }
    return y;
  }

  function graftDotGaussian() -> TransformDot<DelayMultivariateGaussian>? {
    y:TransformDot<DelayMultivariateGaussian>?;
    
    if (y <- left.graftDotGaussian())? {
      y!.divide(right);
    }
    return y;
  }
  
  function graftLinearNormalInverseGamma() ->
      TransformLinear<DelayNormalInverseGamma>? {
    y:TransformLinear<DelayNormalInverseGamma>?;
    z:DelayNormalInverseGamma?;
    
    if (y <- left.graftLinearNormalInverseGamma())? {
      y!.divide(right);
    } else if (z <- left.graftNormalInverseGamma())? {
      y <- TransformLinear<DelayNormalInverseGamma>(1.0/right, z!);
    }
    return y;
  }

  function graftScaledGamma() -> TransformLinear<DelayGamma>? {
    y:TransformLinear<DelayGamma>?;
    z:DelayGamma?;
    
    if (y <- left.graftScaledGamma())? {
      y!.divide(right);
    } else if (z <- left.graftGamma())? {
      y <- TransformLinear<DelayGamma>(1.0/right, z!);
    }
    return y;
  }
}

/**
 * Lazy divide.
 */
operator (left:Expression<Real>/right:Expression<Real>) -> Divide<Real,Real,Real> {
  m:Divide<Real,Real,Real>(left, right);
  return m;
}

/**
 * Lazy divide.
 */
operator (left:Real/right:Expression<Real>) -> Divide<Real,Real,Real> {
  return Boxed(left)/right;
}

/**
 * Lazy divide.
 */
operator (left:Expression<Real>/right:Real) -> Divide<Real,Real,Real> {
  return left/Boxed(right);
}
