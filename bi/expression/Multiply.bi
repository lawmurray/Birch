/**
 * Lazy multiply.
 */
final class Multiply<Left,Right,Value>(left:Expression<Left>,
    right:Expression<Right>) < BinaryExpression<Left,Right,Value>(left, right) {  
  function doValue(l:Left, r:Right) -> Value {
    return l*r;
  }

  function doGradient(d:Value, l:Left, r:Right) -> (Left, Right) {
    return (d*r, d*l);
  }

  function graftScaledGamma() -> TransformLinear<Gamma>? {
    y:TransformLinear<Gamma>?;
    z:Gamma?;
    
    if (y <- left.graftScaledGamma())? {
      y!.multiply(right);
    } else if (y <- right.graftScaledGamma())? {
      y!.multiply(left);
    } else if (z <- left.graftGamma())? {
      y <- TransformLinear<Gamma>(right, z!);
    } else if (z <- right.graftGamma())? {
      y <- TransformLinear<Gamma>(left, z!);
    }
    return y;
  }

  function graftLinearGaussian() -> TransformLinear<Gaussian>? {
    y:TransformLinear<Gaussian>?;
    z:Gaussian?;
    
    if (y <- left.graftLinearGaussian())? {
      y!.multiply(right);
    } else if (y <- right.graftLinearGaussian())? {
      y!.multiply(left);
    } else if (z <- left.graftGaussian())? {
      y <- TransformLinear<Gaussian>(right, z!);
    } else if (z <- right.graftGaussian())? {
      y <- TransformLinear<Gaussian>(left, z!);
    }
    return y;
  }

  function graftDotGaussian() -> TransformDot<MultivariateGaussian>? {
    y:TransformDot<MultivariateGaussian>?;
    
    if (y <- left.graftDotGaussian())? {
      y!.multiply(right);
    } else if (y <- right.graftDotGaussian())? {
      y!.multiply(left);
    }
    return y;
  }
 
  function graftLinearNormalInverseGamma(compare:Distribution<Real>) ->
      TransformLinear<NormalInverseGamma>? {
    y:TransformLinear<NormalInverseGamma>?;
    z:NormalInverseGamma?;
    
    if (y <- left.graftLinearNormalInverseGamma(compare))? {
      y!.multiply(right);
    } else if (y <- right.graftLinearNormalInverseGamma(compare))? {
      y!.multiply(left);
    } else if (z <- left.graftNormalInverseGamma(compare))? {
      y <- TransformLinear<NormalInverseGamma>(right, z!);
    } else if (z <- right.graftNormalInverseGamma(compare))? {
      y <- TransformLinear<NormalInverseGamma>(left, z!);
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
