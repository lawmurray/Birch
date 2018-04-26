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
  
  function isAffineGaussian() -> Boolean {
    return left.isAffineGaussian() || right.isAffineGaussian();
  }
  
  function getAffineGaussian() -> (Real, DelayGaussian, Real) {
    a:Real;
    μ:DelayGaussian?;
    c:Real;
    
    if (left.isAffineGaussian()) {
      (a, μ, c) <- left.getAffineGaussian();
      c <- c + right.value();
    } else if (right.isAffineGaussian()) {
      (a, μ, c) <- right.getAffineGaussian();
      c <- left.value() + c;
    } else {
      assert false;
    }
    return (a, μ!, c);
  }

  function isAffineNormalInverseGamma(σ2:Expression<Real>) -> Boolean {
    return left.isAffineNormalInverseGamma(σ2) ||
        right.isAffineNormalInverseGamma(σ2);
  }
  
  function getAffineNormalInverseGamma(σ2:Expression<Real>) ->
      (Real, DelayNormalInverseGamma, Real) {
    a:Real;
    μ:DelayNormalInverseGamma?;
    c:Real;
    if (left.isAffineNormalInverseGamma(σ2)) {
      (a, μ, c) <- left.getAffineNormalInverseGamma(σ2);
      c <- c + right.value();
    } else if (right.isAffineNormalInverseGamma(σ2)) {
      (a, μ, c) <- right.getAffineNormalInverseGamma(σ2);
      c <- left.value() + c;
    } else {
      assert false;
    }
    return (a, μ!, c);
  }

  function doValue() -> Value {
    return left.value() + right.value();
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
