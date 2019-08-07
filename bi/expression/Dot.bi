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
      TransformMultivariateDotGaussian? {
    y:TransformMultivariateDotGaussian?;
    z:DelayMultivariateGaussian?;
    
    if (z <- right.graftMultivariateGaussian())? {
      y <- TransformMultivariateDotGaussian(left.value(), z!);
    } else if (z <- left.graftMultivariateGaussian())? {
      y <- TransformMultivariateDotGaussian(right.value(), z!);
    }
    return y;
  }
  
  function graftMultivariateDotNormalInverseGamma() ->
      TransformMultivariateDotNormalInverseGamma? {
    y:TransformMultivariateDotNormalInverseGamma?;
    z:DelayMultivariateNormalInverseGamma?;

    if (z <- right.graftMultivariateNormalInverseGamma())? {
      y <- TransformMultivariateDotNormalInverseGamma(left.value(), z!);
    } else if (z <- left.graftMultivariateNormalInverseGamma())? {
      y <- TransformMultivariateDotNormalInverseGamma(right.value(), z!);
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
