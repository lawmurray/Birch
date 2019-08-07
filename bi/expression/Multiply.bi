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

  function graftLinearGaussian() -> TransformLinearGaussian? {
    y:TransformLinearGaussian?;
    z:DelayGaussian?;
    
    if (y <- left.graftLinearGaussian())? {
      y!.multiply(right.value());
    } else if (y <- right.graftLinearGaussian())? {
      y!.multiply(left.value());
    } else if (z <- left.graftGaussian())? {
      y <- TransformLinearGaussian(right.value(), z!, 0.0);
    } else if (z <- right.graftGaussian())? {
      y <- TransformLinearGaussian(left.value(), z!, 0.0);
    }
    return y;
  }
 
  function graftLinearNormalInverseGamma() ->
      TransformLinearNormalInverseGamma? {
    y:TransformLinearNormalInverseGamma?;
    z:DelayNormalInverseGamma?;
    
    if (y <- left.graftLinearNormalInverseGamma())? {
      y!.multiply(right.value());
    } else if (y <- right.graftLinearNormalInverseGamma())? {
      y!.multiply(left.value());
    } else if (z <- left.graftNormalInverseGamma())? {
      y <- TransformLinearNormalInverseGamma(right.value(), z!, 0.0);
    } else if (z <- right.graftNormalInverseGamma())? {
      y <- TransformLinearNormalInverseGamma(left.value(), z!, 0.0);
    }
    return y;
  }

  function graftMultivariateDotGaussian() -> TransformMultivariateDotGaussian? {
    y:TransformMultivariateDotGaussian?;
    
    if (y <- left.graftMultivariateDotGaussian())? {
      y!.multiply(right.value());
    } else if (y <- right.graftMultivariateDotGaussian())? {
      y!.multiply(left.value());
    }
    return y;
  }

  function graftMultivariateDotNormalInverseGamma() ->
      TransformMultivariateDotNormalInverseGamma? {
    y:TransformMultivariateDotNormalInverseGamma?;

    if (y <- left.graftMultivariateDotNormalInverseGamma())? {
      y!.multiply(right.value());
    } else if (y <- right.graftMultivariateDotNormalInverseGamma())? {
      y!.multiply(left.value());
    }
    return y;
  }

  function graftScaledGamma() -> TransformScaledGamma? {
    y:TransformScaledGamma?;
    z:DelayGamma?;
    
    if (y <- left.graftScaledGamma())? {
      y!.multiply(right.value());
    } else if (y <- right.graftScaledGamma())? {
      y!.multiply(left.value());
    } else if (z <- left.graftGamma())? {
      y <- TransformScaledGamma(right.value(), z!);
    } else if (z <- right.graftGamma())? {
      y <- TransformScaledGamma(left.value(), z!);
    }
    return y;
  }
  
  function graftScaledInverseGamma() ->
      TransformScaledInverseGamma? {
    y:TransformScaledInverseGamma?;
    z:DelayInverseGamma?;
    
    if (y <- left.graftScaledInverseGamma())? {
      y!.multiply(right.value());
    } else if (y <- right.graftScaledInverseGamma())? {
      y!.multiply(left.value());
    } else if (z <- left.graftInverseGamma())? {
      y <- TransformScaledInverseGamma(right.value(), z!);        
    } else if (z <- right.graftInverseGamma())? {
      y <- TransformScaledInverseGamma(left.value(), z!);
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
