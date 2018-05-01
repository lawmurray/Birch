/**
 * Delayed addition.
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

  function graftAffineGaussian() -> TransformAffineGaussian? {
    y:TransformAffineGaussian?;
    z:DelayGaussian?;
    
    if (y <- left.graftAffineGaussian())? {
      y!.add(right.value());
    } else if (y <- right.graftAffineGaussian())? {
      y!.add(left.value());
    } else if (z <- left.graftGaussian())? {
      y <- TransformAffineGaussian(1.0, z!, right.value());
    } else if (z <- right.graftGaussian())? {
      y <- TransformAffineGaussian(1.0, z!, left.value());
    }
    return y;
  }
  
  function graftAffineNormalInverseGamma(σ2:Expression<Real>) ->
      TransformAffineNormalInverseGamma? {
    y:TransformAffineNormalInverseGamma?;
    z:DelayNormalInverseGamma?;

    if (y <- left.graftAffineNormalInverseGamma(σ2))? {
      y!.add(right.value());
    } else if (y <- right.graftAffineNormalInverseGamma(σ2))? {
      y!.add(left.value());
    } else if (z <- left.graftNormalInverseGamma(σ2))? {
      y <- TransformAffineNormalInverseGamma(1.0, z!, right.value());
    } else if (z <- right.graftNormalInverseGamma(σ2))? {
      y <- TransformAffineNormalInverseGamma(1.0, z!, left.value());
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
