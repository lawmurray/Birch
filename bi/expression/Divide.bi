/*
 * Lazy division.
 */
final class Divide<Left,Right,Value>(left:Expression<Left>,
    right:Expression<Right>) < Expression<Value> {  
  /**
   * Left operand.
   */
  left:Expression<Left> <- left;
  
  /**
   * Right operand.
   */
  right:Expression<Right> <- right;
  
  function value() -> Value {
    return left.value()/right.value();
  }

  function graftLinearGaussian() -> TransformLinearGaussian? {
    y:TransformLinearGaussian?;
    z:DelayGaussian?;
    
    if (y <- left.graftLinearGaussian())? {
      y!.divide(right.value());
    } else if (z <- left.graftGaussian())? {
      y <- TransformLinearGaussian(1.0/right.value(), z!, 0.0);
    }
    return y;
  }
  
  function graftLinearNormalInverseGamma() -> TransformLinearNormalInverseGamma? {
    y:TransformLinearNormalInverseGamma?;
    z:DelayNormalInverseGamma?;
    
    if (y <- left.graftLinearNormalInverseGamma())? {
      y!.divide(right.value());
    } else if (z <- left.graftNormalInverseGamma())? {
      y <- TransformLinearNormalInverseGamma(1.0/right.value(), z!, 0.0);
    }
    return y;
  }

  function graftMultivariateDotGaussian() -> TransformMultivariateDotGaussian? {
    y:TransformMultivariateDotGaussian?;
    
    if (y <- left.graftMultivariateDotGaussian())? {
      y!.divide(right.value());
    }
    return y;
  }

  function graftMultivariateDotNormalInverseGamma() -> TransformMultivariateDotNormalInverseGamma? {
    y:TransformMultivariateDotNormalInverseGamma?;

    if (y <- left.graftMultivariateDotNormalInverseGamma())? {
      y!.divide(right.value());
    }
    return y;
  }

  function graftScaledGamma() -> TransformScaledGamma? {
    y:TransformScaledGamma?;
    z:DelayGamma?;
    
    if (y <- left.graftScaledGamma())? {
      y!.divide(right.value());
    } else if (z <- left.graftGamma())? {
      y <- TransformScaledGamma(1.0/right.value(), z!);
    }
    return y;
  }
  
  function graftScaledInverseGamma() -> TransformScaledInverseGamma? {
    y:TransformScaledInverseGamma?;
    z:DelayInverseGamma?;
    
    if (y <- left.graftScaledInverseGamma())? {
      y!.divide(right.value());
    } else if (z <- left.graftInverseGamma())? {
      y <- TransformScaledInverseGamma(1.0/right.value(), z!);        
    }
    return y;
  }
}

operator (left:Expression<Real>/right:Expression<Real>) -> Divide<Real,Real,Real> {
  m:Divide<Real,Real,Real>(left, right);
  return m;
}

operator (left:Real/right:Expression<Real>) -> Divide<Real,Real,Real> {
  return Boxed(left)/right;
}

operator (left:Expression<Real>/right:Real) -> Divide<Real,Real,Real> {
  return left/Boxed(right);
}
