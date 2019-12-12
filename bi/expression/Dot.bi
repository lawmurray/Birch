/**
 * Lazy `dot`.
 */
final class Dot<Left,Right,Value>(left:Expression<Left>,
    right:Expression<Right>) < BinaryExpression<Left,Right,Value>(left,
    right) {
  function graft(child:Delay?) -> Expression<Value> {
    return dot(left.graft(child), right.graft(child));
  }

  function doValue(l:Left, r:Right) -> Value {
    return dot(l, r);
  }

  function doGradient(d:Value, l:Left, r:Right) -> (Left, Right) {
    return (d*r, d*l);
  }

  function graftDotGaussian(child:Delay?) -> TransformDot<DelayMultivariateGaussian>? {
    y:TransformLinearMultivariate<DelayMultivariateGaussian>?;
    z:DelayMultivariateGaussian?;
    
    if (y <- right.graftLinearMultivariateGaussian(child))? {
      return TransformDot<DelayMultivariateGaussian>(y!.A*left, y!.x,
          dot(y!.c, left));
    } else if (y <- left.graftLinearMultivariateGaussian(child))? {
      return TransformDot<DelayMultivariateGaussian>(y!.A*right, y!.x,
          dot(y!.c, right));
    } else if (z <- right.graftMultivariateGaussian(child))? {
      return TransformDot<DelayMultivariateGaussian>(left, z!, Boxed(0.0));
    } else if (z <- left.graftMultivariateGaussian(child))? {
      return TransformDot<DelayMultivariateGaussian>(right, z!, Boxed(0.0));
    }
    return nil;
  }
}

/**
 * Lazy `dot`.
 */
function dot(left:Expression<Real[_]>, right:Expression<Real[_]>) ->
    Dot<Real[_],Real[_],Real> {
  assert left.rows() == right.rows();
  m:Dot<Real[_],Real[_],Real>(left, right);
  return m;
}

/**
 * Lazy `dot`.
 */
function dot(left:Real[_], right:Expression<Real[_]>) ->
    Dot<Real[_],Real[_],Real> {
  return dot(Boxed(left), right);
}

/**
 * Lazy `dot`.
 */
function dot(left:Expression<Real[_]>, right:Real[_]) ->
    Dot<Real[_],Real[_],Real> {
  return dot(left, Boxed(right));
}
