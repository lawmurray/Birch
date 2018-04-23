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
  
  function getAffineGaussian() -> (Real, Real, Real, Real) {
    a:Real;
    μ:Real;
    σ2:Real;
    c:Real;
    
    if (left.isAffineGaussian()) {
      (a, μ, σ2, c) <- left.getAffineGaussian();
      a <- a*right.value();
      c <- c*right.value();
    } else if (right.isAffineGaussian()) {
      (a, μ, σ2, c) <- right.getAffineGaussian();
      a <- left.value()*a;
      c <- right.value()*c;
    } else {
      assert false;
    }
    return (a, μ, σ2, c);
  }
  
  function setAffineGaussian(θ:(Real, Real)) {
    if (left.isAffineGaussian()) {
      left.setAffineGaussian(θ);
    } else if (right.isAffineGaussian()) {
      right.setAffineGaussian(θ);
    }
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
