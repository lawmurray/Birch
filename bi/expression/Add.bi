/*
 * Lazy addition.
 */
final class Add<Left,Right,Value>(left:Expression<Left>, right:Expression<Right>) <
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

  function graftLinearGaussian() -> TransformLinear<DelayGaussian>? {
    y:TransformLinear<DelayGaussian>?;
    z:DelayGaussian?;
    
    if (y <- left.graftLinearGaussian())? {
      y!.add(right.value());
    } else if (y <- right.graftLinearGaussian())? {
      y!.add(left.value());
    } else if (z <- left.graftGaussian())? {
      y <- TransformLinear<DelayGaussian>(1.0, z!, right.value());
    } else if (z <- right.graftGaussian())? {
      y <- TransformLinear<DelayGaussian>(1.0, z!, left.value());
    }
    return y;
  }
  
  function graftLinearNormalInverseGamma() ->
      TransformLinear<DelayNormalInverseGamma>? {
    y:TransformLinear<DelayNormalInverseGamma>?;
    z:DelayNormalInverseGamma?;

    if (y <- left.graftLinearNormalInverseGamma())? {
      y!.add(right.value());
    } else if (y <- right.graftLinearNormalInverseGamma())? {
      y!.add(left.value());
    } else if (z <- left.graftNormalInverseGamma())? {
      y <- TransformLinear<DelayNormalInverseGamma>(1.0, z!, right.value());
    } else if (z <- right.graftNormalInverseGamma())? {
      y <- TransformLinear<DelayNormalInverseGamma>(1.0, z!, left.value());
    }
    return y;
  }

  function graftDotMultivariateGaussian() ->
      TransformDot<DelayMultivariateGaussian>? {
    y:TransformDot<DelayMultivariateGaussian>?;
    
    if (y <- left.graftDotMultivariateGaussian())? {
      y!.add(right.value());
    } else if (y <- right.graftDotMultivariateGaussian())? {
      y!.add(left.value());
    }
    return y;
  }

  function graftDotMultivariateNormalInverseGamma() ->
      TransformDot<DelayMultivariateNormalInverseGamma>? {
    y:TransformDot<DelayMultivariateNormalInverseGamma>?;

    if (y <- left.graftDotMultivariateNormalInverseGamma())? {
      y!.add(right.value());
    } else if (y <- right.graftDotMultivariateNormalInverseGamma())? {
      y!.add(left.value());
    }
    return y;
  }

  function graftDiscrete() -> DelayDiscrete? {
    y:DelayDiscrete? <- graftBoundedDiscrete();
    if !y? {
      x:DelayDiscrete?;
      if (x <- left.graftDiscrete())? {
        y <- DelayLinearDiscrete(nil, true, 1, x!, Integer(right.value()));
      } else if (x <- right.graftDiscrete())? {
        y <- DelayLinearDiscrete(nil, true, 1, x!, Integer(left.value()));
      }
    }
    return y;
  }

  function graftBoundedDiscrete() -> DelayBoundedDiscrete? {
    y:DelayBoundedDiscrete?;
    x1:DelayBoundedDiscrete? <- left.graftBoundedDiscrete();
    x2:DelayBoundedDiscrete? <- right.graftBoundedDiscrete();

    if x1? && x2? && !(x1!.hasValue()) {    
      // ^ third condition above ensures that x1 is still valid after x2 is
      //   constructed, which will not be the case if left and right share a
      //   common ancestor on the delayed sampling graph
      y <- DelayAddBoundedDiscrete(nil, true, x1!, x2!);
    } else if x1? && !(x1!.hasValue()) {
      y <- DelayLinearBoundedDiscrete(nil, true, 1, x1!, Integer(right.value()));
    } else if x2? && !(x2!.hasValue()) {
      y <- DelayLinearBoundedDiscrete(nil, true, 1, x2!, Integer(left.value()));
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
