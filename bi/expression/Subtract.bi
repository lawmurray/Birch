/**
 * Lazy subtract.
 */
final class Subtract<Left,Right,Value>(left:Expression<Left>,
    right:Expression<Right>) < BinaryExpression<Left,Right,Value>(left, right) {  
  function graft() -> Expression<Value> {
    return left.graft() - right.graft();
  }

  function doValue(l:Left, r:Right) -> Value {
    return l - r;
  }
  
  function doGradient(d:Value, l:Left, r:Right) -> (Left, Right) {
    return (d, -d);
  }

  function graftLinearGaussian() -> TransformLinear<Gaussian>? {
    y:TransformLinear<Gaussian>?;
    z:Gaussian?;
    
    if (y <- left.graftLinearGaussian())? {
      y!.add(-right);
    } else if (y <- right.graftLinearGaussian())? {
      y!.negateAndAdd(left);
    } else if (z <- left.graftGaussian())? {
      y <- TransformLinear<Gaussian>(Boxed(1.0), z!, -right);
    } else if (z <- right.graftGaussian())? {
      y <- TransformLinear<Gaussian>(Boxed(-1.0), z!, left);
    }
    return y;
  }

  function graftDotGaussian() -> TransformDot<MultivariateGaussian>? {
    y:TransformDot<MultivariateGaussian>?;
    z:Gaussian?;
    
    if (y <- left.graftDotGaussian())? {
      y!.add(-right);
    } else if (y <- right.graftDotGaussian())? {
      y!.negateAndAdd(left);
    }
    return y;
  }

  function graftLinearNormalInverseGamma() ->
      TransformLinear<NormalInverseGamma>? {
    y:TransformLinear<NormalInverseGamma>?;
    z:NormalInverseGamma?;

    if (y <- left.graftLinearNormalInverseGamma())? {
      y!.subtract(right);
    } else if (y <- right.graftLinearNormalInverseGamma())? {
      y!.negateAndAdd(left);
    } else if (z <- left.graftNormalInverseGamma())? {
      y <- TransformLinear<NormalInverseGamma>(Boxed(1.0), z!, -right);
    } else if (z <- right.graftNormalInverseGamma())? {
      y <- TransformLinear<NormalInverseGamma>(Boxed(-1.0), z!, left);
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
