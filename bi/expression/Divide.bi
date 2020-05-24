/**
 * Lazy divide.
 */
final class Divide<Left,Right,Value>(left:Expression<Left>,
    right:Expression<Right>) < BinaryExpression<Left,Right,Value>(left, right) {  
  override function computeValue(l:Left, r:Right) -> Value {
    return l/r;
  }

  override function computeGrad(d:Value, l:Left, r:Right) -> (Left, Right) {
    return (d/r, -d*l/(r*r));
  }

  override function graftLinearGaussian() -> TransformLinear<Gaussian>? {
    y:TransformLinear<Gaussian>?;
    z:Gaussian?;
    
    if (y <- left.graftLinearGaussian())? {
      y!.divide(right);
    } else if (z <- left.graftGaussian())? {
      y <- TransformLinear<Gaussian>(1.0/right, z!);
    }
    return y;
  }

  override function graftDotGaussian() -> TransformDot<MultivariateGaussian>? {
    y:TransformDot<MultivariateGaussian>?;
    
    if (y <- left.graftDotGaussian())? {
      y!.divide(right);
    }
    return y;
  }
  
  override function graftLinearNormalInverseGamma(compare:Distribution<Real>) ->
      TransformLinear<NormalInverseGamma>? {
    y:TransformLinear<NormalInverseGamma>?;
    z:NormalInverseGamma?;
    
    if (y <- left.graftLinearNormalInverseGamma(compare))? {
      y!.divide(right);
    } else if (z <- left.graftNormalInverseGamma(compare))? {
      y <- TransformLinear<NormalInverseGamma>(1.0/right, z!);
    }
    return y;
  }

  override function graftScaledGamma() -> TransformLinear<Gamma>? {
    y:TransformLinear<Gamma>?;
    z:Gamma?;
    
    if (y <- left.graftScaledGamma())? {
      y!.divide(right);
    } else if (z <- left.graftGamma())? {
      y <- TransformLinear<Gamma>(1.0/right, z!);
    }
    return y;
  }
}

/**
 * Lazy divide.
 */
operator (left:Expression<Real>/right:Expression<Real>) -> Expression<Real> {
  if left.isConstant() && right.isConstant() {
    return box(left.value()/right.value());
  } else {
    m:Divide<Real,Real,Real>(left, right);
    return m;
  }
}

/**
 * Lazy divide.
 */
operator (left:Real/right:Expression<Real>) -> Expression<Real> {
  if right.isConstant() {
    return box(left/right.value());
  } else {
    return Boxed(left)/right;
  }
}

/**
 * Lazy divide.
 */
operator (left:Expression<Real>/right:Real) -> Expression<Real> {
  if left.isConstant() {
    return box(left.value()/right);
  } else {
    return left/Boxed(right);
  }
}
