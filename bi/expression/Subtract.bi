/**
 * Lazy subtract.
 */
final class Subtract<Left,Right,Value>(left:Expression<Left>,
    right:Expression<Right>) < BinaryExpression<Left,Right,Value>(left, right) {  
  function doValue(l:Left, r:Right) -> Value {
    return l - r;
  }
  
  function doGradient(d:Value, l:Left, r:Right) -> (Left, Right) {
    return (d, -d);
  }

  function graftLinearGaussian() -> TransformLinear<DelayGaussian>? {
    y:TransformLinear<DelayGaussian>?;
    z:DelayGaussian?;
    
    if (y <- left.graftLinearGaussian())? {
      y!.add(-right.value());
    } else if (y <- right.graftLinearGaussian())? {
      y!.negateAndAdd(left.value());
    } else if (z <- left.graftGaussian())? {
      y <- TransformLinear<DelayGaussian>(1.0, z!, -right.value());
    } else if (z <- right.graftGaussian())? {
      y <- TransformLinear<DelayGaussian>(-1.0, z!, left.value());
    }
    return y;
  }
  
  function graftLinearNormalInverseGamma() ->
      TransformLinear<DelayNormalInverseGamma>? {
    y:TransformLinear<DelayNormalInverseGamma>?;
    z:DelayNormalInverseGamma?;

    if (y <- left.graftLinearNormalInverseGamma())? {
      y!.subtract(right.value());
    } else if (y <- right.graftLinearNormalInverseGamma())? {
      y!.negateAndAdd(left.value());
    } else if (z <- left.graftNormalInverseGamma())? {
      y <- TransformLinear<DelayNormalInverseGamma>(1.0, z!, -right.value());
    } else if (z <- right.graftNormalInverseGamma())? {
      y <- TransformLinear<DelayNormalInverseGamma>(-1.0, z!, left.value());
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

/**
 * Lazy subtract.
 */
operator (left:Expression<Real> - right:Expression<Real>) ->
    Subtract<Real,Real,Real> {
  m:Subtract<Real,Real,Real>(left, right);
  return m;
}

/**
 * Lazy subtract.
 */
operator (left:Real - right:Expression<Real>) -> Subtract<Real,Real,Real> {
  return Boxed(left) - right;
}

/**
 * Lazy subtract.
 */
operator (left:Expression<Real> - right:Real) -> Subtract<Real,Real,Real> {
  return left - Boxed(right);
}

/**
 * Lazy subtract.
 */
operator (left:Expression<Integer> - right:Expression<Integer>) ->
    Subtract<Integer,Integer,Integer> {
  m:Subtract<Integer,Integer,Integer>(left, right);
  return m;
}

/**
 * Lazy subtract.
 */
operator (left:Integer - right:Expression<Integer>) ->
    Subtract<Integer,Integer,Integer> {
  return Boxed(left) - right;
}

/**
 * Lazy subtract.
 */
operator (left:Expression<Integer> - right:Integer) ->
    Subtract<Integer,Integer,Integer> {
  return left - Boxed(right);
}
