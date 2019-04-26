/*
 * Lazy subtraction.
 */
final class Subtract<Left,Right,Value>(left:Expression<Left>,
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

  function graftMultivariateDotGaussian() -> TransformMultivariateDotGaussian? {
    y:TransformMultivariateDotGaussian?;
    
    if (y <- left.graftMultivariateDotGaussian())? {
      y!.add(right.value());
    } else if (y <- right.graftMultivariateDotGaussian())? {
      y!.add(left.value());
    }
    return y;
  }

  function graftMultivariateDotNormalInverseGamma(σ2:Expression<Real>) ->
      TransformMultivariateDotNormalInverseGamma? {
    y:TransformMultivariateDotNormalInverseGamma?;

    if (y <- left.graftMultivariateDotNormalInverseGamma(σ2))? {
      y!.subtract(right.value());
    } else if (y <- right.graftMultivariateDotNormalInverseGamma(σ2))? {
      y!.negateAndAdd(left.value());
    }
    return y;
  }

  function graftDiscrete() -> DelayDiscrete? {
    y:DelayDiscrete? <- graftBoundedDiscrete();
    if (!y?) {
      x:DelayDiscrete?;
      if (x <- left.graftDiscrete())? {
        y <- DelayLinearDiscrete(nil, true, 1, x!, -Integer(right.value()));
      } else if (x <- right.graftDiscrete())? {
        y <- DelayLinearDiscrete(nil, true, -1, x!, Integer(left.value()));
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
      y <- DelaySubtractBoundedDiscrete(nil, true, x1!, x2!);
    } else if x1? && !(x1!.hasValue()) {
      y <- DelayLinearBoundedDiscrete(nil, true, 1, x1!, -Integer(right.value()));
    } else if x2? && !(x2!.hasValue()) {
      y <- DelayLinearBoundedDiscrete(nil, true, -1, x2!, Integer(left.value()));
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
