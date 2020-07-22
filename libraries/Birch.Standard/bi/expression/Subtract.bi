/**
 * Lazy subtract.
 */
final class Subtract(y:Expression<Real>, z:Expression<Real>) <
    ScalarBinaryExpression<Expression<Real>,Expression<Real>,Real,Real,Real,
    Real,Real>(y, z) {
  override function doEvaluate(y:Real, z:Real) -> Real {
    return y - z;
  }
  
  override function doEvaluateGradLeft(d:Real, x:Real, y:Real, z:Real) -> Real {
    return d;
  }

  override function doEvaluateGradRight(d:Real, x:Real, y:Real, z:Real) -> Real {
    return -d;
  }

  override function graftLinearGaussian() -> TransformLinear<Gaussian>? {
    r:TransformLinear<Gaussian>?;
    if !hasValue() {
      x1:Gaussian?;
    
      if (r <- y!.graftLinearGaussian())? {
        r!.add(-z!);
      } else if (r <- z!.graftLinearGaussian())? {
        r!.negateAndAdd(y!);
      } else if (x1 <- y!.graftGaussian())? {
        r <- TransformLinear<Gaussian>(box(1.0), x1!, -z!);
      } else if (x1 <- z!.graftGaussian())? {
        r <- TransformLinear<Gaussian>(box(-1.0), x1!, y!);
      }
    }
    return r;
  }

  override function graftDotGaussian() -> TransformDot<MultivariateGaussian>? {
    r:TransformDot<MultivariateGaussian>?;
    if !hasValue() {
      if (r <- y!.graftDotGaussian())? {
        r!.add(-z!);
      } else if (r <- z!.graftDotGaussian())? {
        r!.negateAndAdd(y!);
      }
    }
    return r;
  }

  override function graftLinearNormalInverseGamma(compare:Distribution<Real>) ->
      TransformLinear<NormalInverseGamma>? {
    r:TransformLinear<NormalInverseGamma>?;
    if !hasValue() {
      x1:NormalInverseGamma?;

      if (r <- y!.graftLinearNormalInverseGamma(compare))? {
        r!.subtract(z!);
      } else if (r <- z!.graftLinearNormalInverseGamma(compare))? {
        r!.negateAndAdd(y!);
      } else if (x1 <- y!.graftNormalInverseGamma(compare))? {
        r <- TransformLinear<NormalInverseGamma>(box(1.0), x1!, -z!);
      } else if (x1 <- z!.graftNormalInverseGamma(compare))? {
        r <- TransformLinear<NormalInverseGamma>(box(-1.0), x1!, y!);
      }
    }
    return r;
  }

  override function graftDotNormalInverseGamma(compare:Distribution<Real>) ->
      TransformDot<MultivariateNormalInverseGamma>? {
    r:TransformDot<MultivariateNormalInverseGamma>?;
    if !hasValue() {
      if (r <- y!.graftDotNormalInverseGamma(compare))? {
        r!.subtract(z!);
      } else if (r <- z!.graftDotNormalInverseGamma(compare))? {
        r!.negateAndAdd(y!);
      }
    }
    return r;
  }
}

/**
 * Lazy subtract.
 */
operator (y:Expression<Real> - z:Expression<Real>) -> Subtract {
  return construct<Subtract>(y, z);
}

/**
 * Lazy subtract.
 */
operator (y:Real - z:Expression<Real>) -> Subtract {
  return box(y) - z;
}

/**
 * Lazy subtract.
 */
operator (y:Expression<Real> - z:Real) -> Subtract {
  return y - box(z);
}
