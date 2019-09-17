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

  function graftDotMultivariateGaussian() ->
      TransformDot<DelayMultivariateGaussian>? {
    y:TransformDot<DelayMultivariateGaussian>?;
    z:DelayMultivariateGaussian?;
    
    if (z <- right.graftMultivariateGaussian())? {
      y <- TransformDot<DelayMultivariateGaussian>(left.value(), z!);
    } else if (z <- left.graftMultivariateGaussian())? {
      y <- TransformDot<DelayMultivariateGaussian>(right.value(), z!);
    }
    return y;
  }
  
  function graftDotIdenticalNormalInverseGamma() ->
      TransformDot<DelayIdenticalNormalInverseGamma>? {
    y:TransformDot<DelayIdenticalNormalInverseGamma>?;
    z:DelayIdenticalNormalInverseGamma?;

    if (z <- right.graftIdenticalNormalInverseGamma())? {
      y <- TransformDot<DelayIdenticalNormalInverseGamma>(left.value(), z!);
    } else if (z <- left.graftIdenticalNormalInverseGamma())? {
      y <- TransformDot<DelayIdenticalNormalInverseGamma>(right.value(), z!);
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
