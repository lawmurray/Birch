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

  function graftLinearGaussian() -> TransformLinear<Gaussian>? {
    y:TransformLinear<Gaussian>?;
    z:Gaussian?;
    
    if (y <- left.graftLinearGaussian())? {
      y!.divide(right);
    } else if (z <- left.graftGaussian())? {
      y <- TransformLinear<Gaussian>(1.0/right, z!);
    }
    return y;
  }

  function graftDotGaussian() -> TransformDot<MultivariateGaussian>? {
    y:TransformDot<MultivariateGaussian>?;
    
    if (y <- left.graftDotGaussian())? {
      y!.divide(right);
    }
    return y;
  }
  
  function graftLinearNormalInverseGamma() ->
      TransformLinear<NormalInverseGamma>? {
    y:TransformLinear<NormalInverseGamma>?;
    z:NormalInverseGamma?;
    
    if (y <- left.graftLinearNormalInverseGamma())? {
      y!.divide(right);
    } else if (z <- left.graftNormalInverseGamma())? {
      y <- TransformLinear<NormalInverseGamma>(1.0/right, z!);
    }
    return y;
  }

  function graftScaledGamma() -> TransformLinear<Gamma>? {
    y:TransformLinear<Gamma>?;
    z:Gamma?;
    
    if (y <- left.graftScaledGamma())? {
      y!.divide(right);
    } else if (z <- left.graftGamma())? {
      y <- TransformLinear<Gamma>(1.0/right, z!);
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
