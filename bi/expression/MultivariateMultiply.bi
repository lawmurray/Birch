/**
 * Delayed multivariate multiplication.
 */
class MultivariateMultiply<Left,Right,Value>(left:Expression<Left>, right:Expression<Right>) < Expression<Value> {  
  /**
   * Left operand.
   */
  left:Expression<Left> <- left;
  
  /**
   * Right operand.
   */
  right:Expression<Right> <- right;
  
  function isMultivariateAffineGaussian() -> Boolean {
    return right.isMultivariateAffineGaussian();
  }
  
  function getMultivariateAffineGaussian() ->
      (Real[_,_], DelayMultivariateGaussian, Real[_]) {    
    assert right.isMultivariateAffineGaussian();
    
    A:Real[_,_];
    μ:DelayMultivariateGaussian?;
    c:Real[_];
    (A, μ, c) <- right.getMultivariateAffineGaussian();
    A <- left.value()*A;
    c <- left.value()*c;

    return (A, μ!, c);
  }

  function isMultivariateScaledInverseGamma(σ2:Expression<Real>) -> Boolean {
    return left.isMultivariateScaledInverseGamma(σ2) || right.isMultivariateScaledInverseGamma(σ2);
  }

  function getMultivariateScaledInverseGamma(σ2:Expression<Real>) ->
      (Real[_,_], DelayInverseGamma) {
    assert right.isMultivariateScaledInverseGamma(σ2);
    
    A:Real[_,_];
    s2:DelayInverseGamma?;
    (A, s2) <- right.getMultivariateScaledInverseGamma(σ2);
    A <- left.value()*A;

    return (A, s2!);
  }
  
  function isMultivariateAffineNormalInverseGamma(σ2:Expression<Real>) -> Boolean {
    return right.isMultivariateAffineNormalInverseGamma(σ2);
  }
  
  function getMultivariateAffineNormalInverseGamma(σ2:Expression<Real>) ->
      (Real[_,_], DelayMultivariateNormalInverseGamma, Real[_]) {
    assert right.isMultivariateAffineNormalInverseGamma(σ2);
    
    A:Real[_,_];
    μ:DelayMultivariateNormalInverseGamma?;
    c:Real[_];
    (A, μ, c) <- right.getMultivariateAffineNormalInverseGamma(σ2);
    A <- left.value()*A;
    c <- left.value()*c;

    return (A, μ!, c);
  }

  function doValue() -> Value {
    return left.value()*right.value();
  }
}

operator (left:Expression<Real[_,_]>*right:Expression<Real[_]>) ->
    MultivariateMultiply<Real[_,_],Real[_],Real[_]> {
  m:MultivariateMultiply<Real[_,_],Real[_],Real[_]>(left, right);
  return m;
}

operator (left:Real[_,_]*right:Expression<Real[_]>) ->
    MultivariateMultiply<Real[_,_],Real[_],Real[_]> {
  return Literal(left)*right;
}

operator (left:Expression<Real[_,_]>*right:Real[_]) ->
    MultivariateMultiply<Real[_,_],Real[_],Real[_]> {
  return left*Literal(right);
}
