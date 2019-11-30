/*
 * Lazy multivariate multiplication.
 */
final class MultivariateMultiply<Left,Right,Value>(left:Expression<Left>,
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
    return left.value()*right.value();
  }

  function pilot() -> Value {
    return left.pilot()*right.pilot();
  }

  function grad(d:Value) {
    left.grad(d*transpose(right.pilot()));
    right.grad(transpose(left.pilot())*d);
  }

  function graftLinearMultivariateGaussian() ->
      TransformLinearMultivariate<DelayMultivariateGaussian>? {
    y:TransformLinearMultivariate<DelayMultivariateGaussian>?;
    z:DelayMultivariateGaussian?;
    
    if (y <- right.graftLinearMultivariateGaussian())? {
      y!.leftMultiply(left.value());
    } else if (z <- right.graftMultivariateGaussian())? {
      y <- TransformLinearMultivariate<DelayMultivariateGaussian>(
          left.value(), z!);
    }
    return y;
  }
  
  function graftLinearMultivariateNormalInverseGamma() ->
      TransformLinearMultivariate<DelayMultivariateNormalInverseGamma>? {
    y:TransformLinearMultivariate<DelayMultivariateNormalInverseGamma>?;
    z:DelayMultivariateNormalInverseGamma?;

    if (y <- right.graftLinearMultivariateNormalInverseGamma())? {
      y!.leftMultiply(left.value());
    } else if (z <- right.graftMultivariateNormalInverseGamma())? {
      y <- TransformLinearMultivariate<DelayMultivariateNormalInverseGamma>(
          left.value(), z!);
    }
    return y;
  }
}

operator (left:Expression<Real[_,_]>*right:Expression<Real[_]>) ->
    MultivariateMultiply<Real[_,_],Real[_],Real[_]> {
  m:MultivariateMultiply<Real[_,_],Real[_],Real[_]>(left, right);
  return m;
}

operator (left:Real[_,_]*right:Expression<Real[_]>) ->
    MultivariateMultiply<Real[_,_],Real[_],Real[_]> {
  return Boxed(left)*right;
}

operator (left:Expression<Real[_,_]>*right:Real[_]) ->
    MultivariateMultiply<Real[_,_],Real[_],Real[_]> {
  return left*Boxed(right);
}
