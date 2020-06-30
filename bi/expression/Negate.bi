/**
 * Lazy negation.
 */
final class Negate(x:Expression<Real>) <
    ScalarUnaryExpression<Expression<Real>,Real>(x) {
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

  override function graftLinearGaussian() -> TransformLinear<Gaussian>? {
    y:TransformLinear<Gaussian>?;
    if !hasValue() {
      z:Gaussian?;
      if (y <- single!.graftLinearGaussian())? {
        y!.negate();
      } else if (z <- single!.graftGaussian())? {
        y <- TransformLinear<Gaussian>(box(-1.0), z!, box(0.0));
      }
    }
    return y;
  }

  override function graftDotGaussian() -> TransformDot<MultivariateGaussian>? {
    y:TransformDot<MultivariateGaussian>?;
    if !hasValue() {
      if (y <- single!.graftDotGaussian())? {
        y!.negate();
      }
    }
    return y;
  }
  
  override function graftLinearNormalInverseGamma(compare:Distribution<Real>) ->
      TransformLinear<NormalInverseGamma>? {
    y:TransformLinear<NormalInverseGamma>?;
    if !hasValue() {
      z:NormalInverseGamma?;
      if (y <- single!.graftLinearNormalInverseGamma(compare))? {
        y!.negate();
      } else if (z <- single!.graftNormalInverseGamma(compare))? {
        y <- TransformLinear<NormalInverseGamma>(box(-1.0), z!, box(0.0));
      }
    }
    return y;
  }

  override function graftDotNormalInverseGamma(compare:Distribution<Real>) ->
      TransformDot<MultivariateNormalInverseGamma>? {
    y:TransformDot<MultivariateNormalInverseGamma>?;
    if !hasValue() {
      if (y <- single!.graftDotNormalInverseGamma(compare))? {
        y!.negate();
      }
    }
    return y;
  }
}

/**
 * Lazy negation.
 */
operator (-x:Expression<Real>) -> Expression<Real> {
  if x.isConstant() {
    return box(-x.value());
  } else {
    m:Negate(x);
    return m;
  }
}
