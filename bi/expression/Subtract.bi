/**
 * Lazy subtraction.
 */
class Subtract<Left,Right,Value>(left:Expression<Left>,
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
    return left.value() - right.value();
  }

  function graftLinearGaussian() -> TransformLinearGaussian? {
    y:TransformLinearGaussian?;
    z:DelayGaussian?;
    
    if (y <- left.graftLinearGaussian())? {
      y!.add(-right.value());
    } else if (y <- right.graftLinearGaussian())? {
      y!.negateAndAdd(left.value());
    } else if (z <- left.graftGaussian())? {
      y <- TransformLinearGaussian(1.0, z!, -right.value());
    } else if (z <- right.graftGaussian())? {
      y <- TransformLinearGaussian(-1.0, z!, left.value());
    }
    return y;
  }
  
  function graftLinearNormalInverseGamma(σ2:Expression<Real>) ->
      TransformLinearNormalInverseGamma? {
    y:TransformLinearNormalInverseGamma?;
    z:DelayNormalInverseGamma?;

    if (y <- left.graftLinearNormalInverseGamma(σ2))? {
      y!.subtract(right.value());
    } else if (y <- right.graftLinearNormalInverseGamma(σ2))? {
      y!.negateAndAdd(left.value());
    } else if (z <- left.graftNormalInverseGamma(σ2))? {
      y <- TransformLinearNormalInverseGamma(1.0, z!, -right.value());
    } else if (z <- right.graftNormalInverseGamma(σ2))? {
      y <- TransformLinearNormalInverseGamma(-1.0, z!, left.value());
    }
    return y;
  }

  function graftLinearBinomial() -> TransformLinearBinomial? {
    y:TransformLinearBinomial?;
    z:DelayBinomial?;
    
    if (y <- left.graftLinearBinomial())? {
      y!.subtract(Integer(right.value()));
    } else if (y <- right.graftLinearBinomial())? {
      y!.negateAndAdd(Integer(left.value()));
    } else if (z <- left.graftBinomial())? {
      y <- TransformLinearBinomial(1, z!, -Integer(right.value()));
    } else if (z <- right.graftBinomial())? {
      y <- TransformLinearBinomial(-1, z!, Integer(left.value()));
    }
    return y;
  }
}

operator (left:Expression<Real> - right:Expression<Real>) ->
    Subtract<Real,Real,Real> {
  m:Subtract<Real,Real,Real>(left, right);
  return m;
}

operator (left:Real - right:Expression<Real>) -> Subtract<Real,Real,Real> {
  return Boxed(left) - right;
}

operator (left:Expression<Real> - right:Real) -> Subtract<Real,Real,Real> {
  return left - Boxed(right);
}

operator (left:Expression<Integer> - right:Expression<Integer>) ->
    Subtract<Integer,Integer,Integer> {
  m:Subtract<Integer,Integer,Integer>(left, right);
  return m;
}

operator (left:Integer - right:Expression<Integer>) ->
    Subtract<Integer,Integer,Integer> {
  return Boxed(left) - right;
}

operator (left:Expression<Integer> - right:Integer) ->
    Subtract<Integer,Integer,Integer> {
  return left - Boxed(right);
}
