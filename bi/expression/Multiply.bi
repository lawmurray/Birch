/*
 * Lazy multiplication.
 */
final class Multiply<Left,Right,Value>(left:Expression<Left>, right:Expression<Right>) < Expression<Value> {  
  /**
   * Left operand.
   */
  left:Expression<Left> <- left;
  
  /**
   * Right operand.
   */
  right:Expression<Right> <- right;
  
  function value() -> Value {
    return left.value()*right.value();
  }

  function pilot() -> Value {
    return left.pilot()*right.pilot();
  }

  function graftScaledGamma() -> TransformLinear<DelayGamma>? {
    y:TransformLinear<DelayGamma>?;
    z:DelayGamma?;
    
    if (y <- left.graftScaledGamma())? {
      y!.multiply(right.value());
    } else if (y <- right.graftScaledGamma())? {
      y!.multiply(left.value());
    } else if (z <- left.graftGamma())? {
      y <- TransformLinear<DelayGamma>(right.value(), z!);
    } else if (z <- right.graftGamma())? {
      y <- TransformLinear<DelayGamma>(left.value(), z!);
    }
    return y;
  }

  function graftLinearGaussian() -> TransformLinear<DelayGaussian>? {
    y:TransformLinear<DelayGaussian>?;
    z:DelayGaussian?;
    
    if (y <- left.graftLinearGaussian())? {
      y!.multiply(right.value());
    } else if (y <- right.graftLinearGaussian())? {
      y!.multiply(left.value());
    } else if (z <- left.graftGaussian())? {
      y <- TransformLinear<DelayGaussian>(right.value(), z!);
    } else if (z <- right.graftGaussian())? {
      y <- TransformLinear<DelayGaussian>(left.value(), z!);
    }
    return y;
  }
 
  function graftLinearNormalInverseGamma() ->
      TransformLinear<DelayNormalInverseGamma>? {
    y:TransformLinear<DelayNormalInverseGamma>?;
    z:DelayNormalInverseGamma?;
    
    if (y <- left.graftLinearNormalInverseGamma())? {
      y!.multiply(right.value());
    } else if (y <- right.graftLinearNormalInverseGamma())? {
      y!.multiply(left.value());
    } else if (z <- left.graftNormalInverseGamma())? {
      y <- TransformLinear<DelayNormalInverseGamma>(right.value(), z!);
    } else if (z <- right.graftNormalInverseGamma())? {
      y <- TransformLinear<DelayNormalInverseGamma>(left.value(), z!);
    }
    return y;
  }
}

operator (left:Expression<Real>*right:Expression<Real>) ->
    Multiply<Real,Real,Real> {
  m:Multiply<Real,Real,Real>(left, right);
  return m;
}

operator (left:Real*right:Expression<Real>) -> Multiply<Real,Real,Real> {
  return Boxed(left)*right;
}

operator (left:Expression<Real>*right:Real) -> Multiply<Real,Real,Real> {
  return left*Boxed(right);
}

operator (left:Expression<Integer>*right:Expression<Integer>) ->
    Multiply<Integer,Integer,Integer> {
  m:Multiply<Integer,Integer,Integer>(left, right);
  return m;
}

operator (left:Integer*right:Expression<Integer>) ->
    Multiply<Integer,Integer,Integer> {
  return Boxed(left)*right;
}

operator (left:Expression<Integer>*right:Integer) ->
    Multiply<Integer,Integer,Integer> {
  return left*Boxed(right);
}
