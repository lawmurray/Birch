/**
 * Delayed multiplication.
 */
class Multiply<Left,Right,Value>(left:Expression<Left>, right:Expression<Right>) < Expression<Value> {  
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
      a <- a*right.value();
      c <- c*right.value();
    } else if (right.isAffineGaussian()) {
      (a, μ, c) <- right.getAffineGaussian();
      a <- left.value()*a;
      c <- left.value()*c;
    } else {
      assert false;
    }
    return (a, μ!, c);
  }

  function isScaledInverseGamma(σ2:Expression<Real>) -> Boolean {
    return left.isScaledInverseGamma(σ2) || right.isScaledInverseGamma(σ2);
  }

  function getScaledInverseGamma(σ2:Expression<Real>) -> (Real, Real, Real) {
    a2:Real;
    α:Real;
    β:Real;
    if (left.isScaledInverseGamma(σ2)) {
      (a2, α, β) <- left.getScaledInverseGamma(σ2);
      a2 <- a2*right.value();
    } else if (right.isScaledInverseGamma(σ2)) {
      (a2, α, β) <- right.getScaledInverseGamma(σ2);
      a2 <- left.value()*a2;
    }
    return (a2, α, β);
  }
  
  function isAffineNormalInverseGamma(σ2:Expression<Real>) -> Boolean {
    return left.isAffineNormalInverseGamma(σ2) || right.isAffineNormalInverseGamma(σ2);
  }
  
  function getAffineNormalInverseGamma(σ2:Expression<Real>) -> (Real, Real, Real, Real, Real, Real) {
    a:Real;
    μ:Real;
    c:Real;
    a2:Real;
    α:Real;
    β:Real;
    if (left.isAffineNormalInverseGamma(σ2)) {
      (a, μ, c, a2, α, β) <- left.getAffineNormalInverseGamma(σ2);
      a <- a*right.value();
      c <- c*right.value();
    } else if (right.isAffineNormalInverseGamma(σ2)) {
      (a, μ, c, a2, α, β) <- right.getAffineNormalInverseGamma(σ2);
      a <- left.value()*a;
      c <- left.value()*c;
    } else {
      assert false;
    }
    return (a, μ, c, a2, α, β);
  }

  function doValue() -> Value {
    return left.value()*right.value();
  }
}

operator (left:Expression<Real>*right:Expression<Real>) -> Multiply<Real,Real,Real> {
  m:Multiply<Real,Real,Real>(left, right);
  return m;
}

operator (left:Real*right:Expression<Real>) -> Multiply<Real,Real,Real> {
  return Literal(left)*right;
}

operator (left:Expression<Real>*right:Real) -> Multiply<Real,Real,Real> {
  return left*Literal(right);
}
