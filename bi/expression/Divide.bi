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

  function graftLinearGaussian() -> TransformLinear<DelayGaussian>? {
    y:TransformLinear<DelayGaussian>?;
    z:DelayGaussian?;
    
    if (y <- left.graftLinearGaussian())? {
      y!.divide(right.value());
    } else if (z <- left.graftGaussian())? {
      y <- TransformLinear<DelayGaussian>(1.0/right.value(), z!);
    }
    return y;
  }
  
  function graftLinearNormalInverseGamma() ->
      TransformLinear<DelayNormalInverseGamma>? {
    y:TransformLinear<DelayNormalInverseGamma>?;
    z:DelayNormalInverseGamma?;
    
    if (y <- left.graftLinearNormalInverseGamma())? {
      y!.divide(right.value());
    } else if (z <- left.graftNormalInverseGamma())? {
      y <- TransformLinear<DelayNormalInverseGamma>(1.0/right.value(), z!);
    }
    return y;
  }

  function graftScaledGamma() -> TransformLinear<DelayGamma>? {
    y:TransformLinear<DelayGamma>?;
    z:DelayGamma?;
    
    if (y <- left.graftScaledGamma())? {
      y!.divide(right.value());
    } else if (z <- left.graftGamma())? {
      y <- TransformLinear<DelayGamma>(1.0/right.value(), z!);
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
