/**
 * Delayed multivariation addition.
 */
class MultivariateAdd<Left,Right,Value>(left:Expression<Left>, right:Expression<Right>) <
    Expression<Value> {  
  /**
   * Left operand.
   */
  left:Expression<Left> <- left;
  
  /**
   * Right operand.
   */
  right:Expression<Right> <- right;
  
  function isMultivariateAffineGaussian() -> Boolean {
    return left.isMultivariateAffineGaussian() ||
        right.isMultivariateAffineGaussian();
  }
  
  function getMultivariateAffineGaussian() -> (Real[_,_], DelayMultivariateGaussian, Real[_]) {
    A:Real[_,_];
    μ:DelayMultivariateGaussian?;
    c:Real[_];
    
    if (left.isMultivariateAffineGaussian()) {
      (A, μ, c) <- left.getMultivariateAffineGaussian();
      c <- c + right.value();
    } else if (right.isMultivariateAffineGaussian()) {
      (A, μ, c) <- right.getMultivariateAffineGaussian();
      c <- left.value() + c;
    } else {
      assert false;
    }
    return (A, μ!, c);
  }

  function isMultivariateAffineNormalInverseGamma(σ2:Expression<Real>) -> Boolean {
    return left.isMultivariateAffineNormalInverseGamma(σ2) ||
        right.isMultivariateAffineNormalInverseGamma(σ2);
  }
  
  function getMultivariateAffineNormalInverseGamma(σ2:Expression<Real>) ->
      (Real[_,_], DelayMultivariateNormalInverseGamma, Real[_]) {
    A:Real[_,_];
    μ:DelayMultivariateNormalInverseGamma?;
    c:Real[_];
    if (left.isMultivariateAffineNormalInverseGamma(σ2)) {
      (A, μ, c) <- left.getMultivariateAffineNormalInverseGamma(σ2);
      c <- c + right.value();
    } else if (right.isMultivariateAffineNormalInverseGamma(σ2)) {
      (A, μ, c) <- right.getMultivariateAffineNormalInverseGamma(σ2);
      c <- left.value() + c;
    } else {
      assert false;
    }
    return (A, μ!, c);
  }

  function doValue() -> Value {
    return left.value() + right.value();
  }
}

operator (left:Expression<Real[_]> + right:Expression<Real[_]>) ->
    MultivariateAdd<Real[_],Real[_],Real[_]> {
  m:MultivariateAdd<Real[_],Real[_],Real[_]>(left, right);
  return m;
}

operator (left:Real[_] + right:Expression<Real[_]>) ->
    MultivariateAdd<Real[_],Real[_],Real[_]> {
  return Literal(left) + right;
}

operator (left:Expression<Real[_]> + right:Real[_]) ->
    MultivariateAdd<Real[_],Real[_],Real[_]> {
  return left + Literal(right);
}
