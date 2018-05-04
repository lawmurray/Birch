/**
 * Lazy division.
 */
class Divide<Left,Right,Value>(left:Expression<Left>,
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
  
  function getLinearNormalInverseGamma(σ2:Expression<Real>) ->
      TransformLinearNormalInverseGamma? {
    y:TransformLinearNormalInverseGamma?;
    z:DelayNormalInverseGamma?;
    
    if (y <- left.graftLinearNormalInverseGamma(σ2))? {
      y!.divide(right.value());
    } else if (z <- left.graftNormalInverseGamma(σ2))? {
      y <- TransformLinearNormalInverseGamma(1.0/right.value(), z!, 0.0);
    }
    return y;
  }
  
  function graftScaledInverseGamma(σ2:Expression<Real>) ->
      TransformScaledInverseGamma? {
    y:TransformScaledInverseGamma?;
    z:DelayInverseGamma?;
    
    if (y <- left.graftScaledInverseGamma(σ2))? {
      y!.divide(right.value());
    } else if Object(left) == σ2 && (z <- left.graftInverseGamma())? {
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
