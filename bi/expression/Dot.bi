/**
 * Lazy `dot`.
 */
final class Dot<Left,Right,Value>(left:Expression<Left>,
    right:Expression<Right>) < BinaryExpression<Left,Right,Value>(left,
    right) {
  function doValue(l:Left, r:Right) -> Value {
    return dot(l, r);
  }

  function doGradient(d:Value, l:Left, r:Right) -> (Left, Right) {
    return (d*r, d*l);
  }

  function graftDotGaussian() -> TransformDot<DelayMultivariateGaussian>? {
    y:TransformLinearMultivariate<DelayMultivariateGaussian>?;
    z:DelayMultivariateGaussian?;
    
    if (y <- right.graftLinearMultivariateGaussian())? {
      return TransformDot<DelayMultivariateGaussian>(y!.A*left.value(), y!.x,
          dot(y!.c, left.value()));
    } else if (y <- left.graftLinearMultivariateGaussian())? {
      return TransformDot<DelayMultivariateGaussian>(y!.A*right.value(), y!.x,
          dot(y!.c, right.value()));
    } else if (z <- right.graftMultivariateGaussian())? {
      return TransformDot<DelayMultivariateGaussian>(left.value(), z!, 0.0);
    } else if (z <- left.graftMultivariateGaussian())? {
      return TransformDot<DelayMultivariateGaussian>(right.value(), z!, 0.0);
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
