/**
 * Lazy negation.
 */
final class MultivariateNegate(x:Expression<Real[_]>) <
    ScalarUnaryExpression<Expression<Real[_]>,Real[_]>(x) {
  override function doValue() {
    x <- -single!.value();
  }

  override function doPilot() {
    x <- -single!.pilot();
  }

  override function doMove(κ:Kernel) {
    x <- -single!.move(κ);
  }

  override function doGrad() {
    single!.grad(-d!);
  }

  override function graftLinearMultivariateGaussian() ->
      TransformLinearMultivariate<MultivariateGaussian>? {
    y:TransformLinearMultivariate<MultivariateGaussian>?;
    if !hasValue() {
      z:MultivariateGaussian?;

      if (y <- single!.graftLinearMultivariateGaussian())? {
        y!.negate();
      } else if (z <- single!.graftMultivariateGaussian())? {
        auto R <- z!.rows();
        y <- TransformLinearMultivariate<MultivariateGaussian>(
            box(diagonal(-1.0, R)), z!, box(vector(0.0, R)));
      }
    }
    return y;
  }
  
  override function graftLinearMultivariateNormalInverseGamma(compare:Distribution<Real>) ->
      TransformLinearMultivariate<MultivariateNormalInverseGamma>? {
    y:TransformLinearMultivariate<MultivariateNormalInverseGamma>?;
    if !hasValue() {
      z:MultivariateNormalInverseGamma?;

      if (y <- single!.graftLinearMultivariateNormalInverseGamma(compare))? {
        y!.negate();
      } else if (z <- single!.graftMultivariateNormalInverseGamma(compare))? {
        auto R <- z!.rows();
        y <- TransformLinearMultivariate<MultivariateNormalInverseGamma>(
            box(diagonal(-1.0, R)), z!, box(vector(0.0, R)));
      }
    }
    return y;
  }
}

/**
 * Lazy negation.
 */
operator (-x:Expression<Real[_]>) -> Expression<Real[_]> {
  if x.isConstant() {
    return box(vector(-x.value()));
  } else {
    m:MultivariateNegate(x);
    return m;
  }
}
