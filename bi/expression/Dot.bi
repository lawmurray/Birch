/**
 * Lazy `dot`.
 */
final class Dot<Left,Right,Value>(left:Left, right:Right) <
    ScalarBinaryExpression<Left,Right,Value>(left, right) {
  override function doValue() {
    x <- dot(left.value(), right.value());
  }

  override function doPilot() {
    x <- dot(left.pilot(), right.pilot());
  }

  override function doMove(κ:Kernel) {
    x <- dot(left.move(κ), right.move(κ));
  }

  override function doGrad() {
    left.grad(d!*right.get());
    right.grad(d!*left.get());
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
      return TransformDot<MultivariateGaussian>(left, z!, box(0.0));
    } else if (z <- left.graftMultivariateGaussian())? {
      return TransformDot<MultivariateGaussian>(right, z!, box(0.0));
    }
    return nil;
  }

  override function graftDotNormalInverseGamma(compare:Distribution<Real>) ->
      TransformDot<MultivariateNormalInverseGamma>? {
    y:TransformLinearMultivariate<MultivariateNormalInverseGamma>?;
    z:MultivariateNormalInverseGamma?;
    
    if (y <- right.graftLinearMultivariateNormalInverseGamma(compare))? {
      return TransformDot<MultivariateNormalInverseGamma>(
          transpose(y!.A)*left, y!.x, dot(left, y!.c));
    } else if (y <- left.graftLinearMultivariateNormalInverseGamma(compare))? {
      return TransformDot<MultivariateNormalInverseGamma>(y!.A*right, y!.x,
          dot(y!.c, right));
    } else if (z <- right.graftMultivariateNormalInverseGamma(compare))? {
      return TransformDot<MultivariateNormalInverseGamma>(left, z!, box(0.0));
    } else if (z <- left.graftMultivariateNormalInverseGamma(compare))? {
      return TransformDot<MultivariateNormalInverseGamma>(right, z!, box(0.0));
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
    m:Dot<Expression<Real[_]>,Expression<Real[_]>,Real>(left, right);
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
    return dot(box(left), right);
  }
}

/**
 * Lazy `dot`.
 */
function dot(left:Expression<Real[_]>, right:Real[_]) -> Expression<Real> {
  if left.isConstant() {
    return box(dot(left.value(), right));
  } else {
    return dot(left, box(right));
  }
}
