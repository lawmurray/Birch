/**
 * Lazy `dot`.
 */
final class Dot<Left,Right,Value>(left:Expression<Left>,
    right:Expression<Right>) < BinaryExpression<Left,Right,Value>(left,
    right) {
  override function computeValue(l:Left, r:Right) -> Value {
    return dot(l, r);
  }

  override function computeGrad(d:Value, l:Left, r:Right) -> (Left, Right) {
    return (d*r, d*l);
  }

  override function graftDotGaussian() -> TransformDot<MultivariateGaussian>? {
    y:TransformLinearMultivariate<MultivariateGaussian>?;
    z:MultivariateGaussian?;
    
    if (y <- right.graftLinearMultivariateGaussian())? {
      return TransformDot<MultivariateGaussian>(y!.A*left, y!.x,
          dot(y!.c, left));
    } else if (y <- left.graftLinearMultivariateGaussian())? {
      return TransformDot<MultivariateGaussian>(y!.A*right, y!.x,
          dot(y!.c, right));
    } else if (z <- right.graftMultivariateGaussian())? {
      return TransformDot<MultivariateGaussian>(left, z!, Boxed(0.0));
    } else if (z <- left.graftMultivariateGaussian())? {
      return TransformDot<MultivariateGaussian>(right, z!, Boxed(0.0));
    }
    return nil;
  }
}

/**
 * Lazy `dot`.
 */
function dot(left:Expression<Real[_]>, right:Expression<Real[_]>) ->
    Expression<Real> {
  assert left.rows() == right.rows();
  if left.isConstant() && right.isConstant() {
    return box(dot(left.value(), right.value()));
  } else {
    m:Dot<Real[_],Real[_],Real>(left, right);
    return m;
  }
}

/**
 * Lazy `dot`.
 */
function dot(left:Real[_], right:Expression<Real[_]>) -> Expression<Real> {
  if right.isConstant() {
    return box(dot(left, right.value()));
  } else {
    return dot(Boxed(left), right);
  }
}

/**
 * Lazy `dot`.
 */
function dot(left:Expression<Real[_]>, right:Real[_]) -> Expression<Real> {
  if left.isConstant() {
    return box(dot(left.value(), right));
  } else {
    return dot(left, Boxed(right));
  }
}
