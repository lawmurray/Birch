/**
 * Delayed multiplication.
 */
class Multiply<Left,Right,Value>(left:Expression<Left>, right:Expression<Right>) < Expression<Value> {  
  /**
   * Left operand.
   */
  left:Expression<Left> <- left;
  
  /**
   * Right operand.
   */
  right:Expression<Right> <- right;
  
  function graftAffineGaussian() -> TransformAffineGaussian? {
    y:TransformAffineGaussian?;
    z:DelayGaussian?;
    
    if (y <- left.graftAffineGaussian())? {
      y!.multiply(right.value());
    } else if (y <- right.graftAffineGaussian())? {
      y!.multiply(left.value());
    } else if (z <- left.graftGaussian())? {
      y <- TransformAffineGaussian(right.value(), z!, 0.0);
    } else if (z <- right.graftGaussian())? {
      y <- TransformAffineGaussian(left.value(), z!, 0.0);
    }
    return y;
  }
  
  function getAffineNormalInverseGamma(σ2:Expression<Real>) ->
      TransformAffineNormalInverseGamma? {
    y:TransformAffineNormalInverseGamma?;
    z:DelayNormalInverseGamma?;
    
    if (y <- left.graftAffineNormalInverseGamma(σ2))? {
      y!.multiply(right.value());
    } else if (y <- right.graftAffineNormalInverseGamma(σ2))? {
      y!.multiply(left.value());
    } else if (z <- left.graftNormalInverseGamma(σ2))? {
      y <- TransformAffineNormalInverseGamma(right.value(), z!, 0.0);
    } else if (z <- right.graftNormalInverseGamma(σ2))? {
      y <- TransformAffineNormalInverseGamma(left.value(), z!, 0.0);
    }
    return y;
  }
  
  function graftScaledInverseGamma(σ2:DelayInverseGamma) ->
      TransformScaledInverseGamma? {
    y:TransformScaledInverseGamma?;
    z:DelayInverseGamma?;
    
    if (y <- left.graftScaledInverseGamma(σ2))? {
      y!.multiply(right.value());
    } else if (y <- right.graftScaledInverseGamma(σ2))? {
      y!.multiply(left.value());
    } else if (z <- left.graftInverseGamma(σ2))? {
      y <- TransformScaledInverseGamma(right.value(), z!);        
    } else if (z <- right.graftInverseGamma(σ2))? {
      y <- TransformScaledInverseGamma(left.value(), z!);
    }
    return y;
  }

  function doValue() -> Value {
    return left.value()*right.value();
  }
}

operator (left:Expression<Real>*right:Expression<Real>) -> Multiply<Real,Real,Real> {
  m:Multiply<Real,Real,Real>(left, right);
  return m;
}

operator (left:Real*right:Expression<Real>) -> Multiply<Real,Real,Real> {
  return Boxed(left)*right;
}

operator (left:Expression<Real>*right:Real) -> Multiply<Real,Real,Real> {
  return left*Boxed(right);
}
