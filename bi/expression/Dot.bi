/*
 * Lazy multivariate multiplication.
 */
final class Dot<Left,Right,Value>(left:Expression<Left>, right:Expression<Right>) <
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
    return dot(left.value(), right.value());
  }

  function graftMultivariateDotGaussian() ->
      TransformDotMultivariateGaussian? {
    y:TransformDotMultivariateGaussian?;
    z:DelayMultivariateGaussian?;
    
    if (z <- right.graftMultivariateGaussian())? {
      y <- TransformDotMultivariateGaussian(left.value(), z!);
    } else if (z <- left.graftMultivariateGaussian())? {
      y <- TransformDotMultivariateGaussian(right.value(), z!);
    }
    return y;
  }
  
  function graftMultivariateDotNormalInverseGamma() ->
      TransformDotIdenticalNormalInverseGamma? {
    y:TransformDotIdenticalNormalInverseGamma?;
    z:DelayIdenticalNormalInverseGamma?;

    if (z <- right.graftIdenticalNormalInverseGamma())? {
      y <- TransformDotIdenticalNormalInverseGamma(left.value(), z!);
    } else if (z <- left.graftIdenticalNormalInverseGamma())? {
      y <- TransformDotIdenticalNormalInverseGamma(right.value(), z!);
    }
    return y;
  }
}

function dot(left:Expression<Real[_]>, right:Expression<Real[_]>) ->
    Dot<Real[_],Real[_],Real> {
  m:Dot<Real[_],Real[_],Real>(left, right);
  return m;
}

function dot(left:Real[_], right:Expression<Real[_]>) ->
    Dot<Real[_],Real[_],Real> {
  return dot(Boxed(left), right);
}

function dot(left:Expression<Real[_]>, right:Real[_]) ->
    Dot<Real[_],Real[_],Real> {
  return dot(left, Boxed(right));
}
