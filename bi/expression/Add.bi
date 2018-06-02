/*
 * Lazy addition.
 */
class Add<Left,Right,Value>(left:Expression<Left>, right:Expression<Right>) <
    Expression<Value> {  
  /**
   * Left operand.
   */
  left:Expression<Left> <- left;
  
  /**
   * Right operand.
   */
  right:Expression<Right> <- right;

  function value() -> Value {
    return left.value() + right.value();
  }

  function graftLinearGaussian() -> TransformLinearGaussian? {
    y:TransformLinearGaussian?;
    z:DelayGaussian?;
    
    if (y <- left.graftLinearGaussian())? {
      y!.add(right.value());
    } else if (y <- right.graftLinearGaussian())? {
      y!.add(left.value());
    } else if (z <- left.graftGaussian())? {
      y <- TransformLinearGaussian(1.0, z!, right.value());
    } else if (z <- right.graftGaussian())? {
      y <- TransformLinearGaussian(1.0, z!, left.value());
    }
    return y;
  }
  
  function graftLinearNormalInverseGamma(σ2:Expression<Real>) ->
      TransformLinearNormalInverseGamma? {
    y:TransformLinearNormalInverseGamma?;
    z:DelayNormalInverseGamma?;

    if (y <- left.graftLinearNormalInverseGamma(σ2))? {
      y!.add(right.value());
    } else if (y <- right.graftLinearNormalInverseGamma(σ2))? {
      y!.add(left.value());
    } else if (z <- left.graftNormalInverseGamma(σ2))? {
      y <- TransformLinearNormalInverseGamma(1.0, z!, right.value());
    } else if (z <- right.graftNormalInverseGamma(σ2))? {
      y <- TransformLinearNormalInverseGamma(1.0, z!, left.value());
    }
    return y;
  }

  function graftLinearDiscrete() -> TransformLinearDiscrete? {
    y:TransformLinearDiscrete?;
    z:DelayDiscrete?;
    
    if (y <- left.graftLinearDiscrete())? {
      y!.add(Integer(right.value()));
    } else if (y <- right.graftLinearDiscrete())? {
      y!.add(Integer(left.value()));
    } else if (z <- left.graftDiscrete())? {
      y <- TransformLinearDiscrete(1, z!, Integer(right.value()));
    } else if (z <- right.graftDiscrete())? {
      y <- TransformLinearDiscrete(1, z!, Integer(left.value()));
    }
    return y;
  }

  function graftAddBoundedDiscrete() -> TransformAddBoundedDiscrete? {
    y:TransformAddBoundedDiscrete?;
    x1:DelayBoundedDiscrete?;
    x2:DelayBoundedDiscrete?;
    
    if (x1 <- left.graftBoundedDiscrete())? &&
        (x2 <- right.graftBoundedDiscrete())? &&
        (left.graftBoundedDiscrete())? {
      // ^ third condition above ensures that x1 is still valid after x2 is
      //   constructed, which will not be the case if left and right share a
      //   common ancestor on the delayed sampling graph
      y <- TransformAddBoundedDiscrete(x1!, x2!);
    }
    return y;
  }
}

operator (left:Expression<Real> + right:Expression<Real>) ->
    Add<Real,Real,Real> {
  m:Add<Real,Real,Real>(left, right);
  return m;
}

operator (left:Real + right:Expression<Real>) -> Add<Real,Real,Real> {
  return Boxed(left) + right;
}

operator (left:Expression<Real> + right:Real) -> Add<Real,Real,Real> {
  return left + Boxed(right);
}

operator (left:Expression<Integer> + right:Expression<Integer>) ->
    Add<Integer,Integer,Integer> {
  m:Add<Integer,Integer,Integer>(left, right);
  return m;
}

operator (left:Integer + right:Expression<Integer>) ->
    Add<Integer,Integer,Integer> {
  return Boxed(left) + right;
}

operator (left:Expression<Integer> + right:Integer) ->
    Add<Integer,Integer,Integer> {
  return left + Boxed(right);
}
