/**
 * Lazy add.
 */
final class Add(y:Expression<Real>, z:Expression<Real>) <
    ScalarBinaryExpression<Expression<Real>,Expression<Real>,Real,Real,Real,
    Real,Real>(y, z) {  
  override function doEvaluate(y:Real, z:Real) -> Real {
    return y + z;
  }
  
  override function doEvaluateGradLeft(d:Real, x:Real, y:Real,
      z:Real) -> Real {
    return d;
  }

  override function doEvaluateGradRight(d:Real, x:Real, y:Real,
      z:Real) -> Real {
    return d;
  }

  override function graftLinearGaussian() -> TransformLinear<Gaussian>? {
    r:TransformLinear<Gaussian>?;
    if !hasValue() {
      x1:Gaussian?;
    
      if (r <- y!.graftLinearGaussian())? {
        r!.add(z!);
      } else if (r <- z!.graftLinearGaussian())? {
        r!.add(y!);
      } else if (x1 <- y!.graftGaussian())? {
        r <- TransformLinear<Gaussian>(box(1.0), x1!, z!);
      } else if (x1 <- z!.graftGaussian())? {
        r <- TransformLinear<Gaussian>(box(1.0), x1!, y!);
      }
    }
    return r;
  }

  override function graftDotGaussian() -> TransformDot<MultivariateGaussian>? {
    r:TransformDot<MultivariateGaussian>?;
    if !hasValue() {
      if (r <- y!.graftDotGaussian())? {
        r!.add(z!);
      } else if (r <- z!.graftDotGaussian())? {
        r!.add(y!);
      }
    }
    return r;
  }
  
  override function graftLinearNormalInverseGamma(compare:Distribution<Real>) ->
      TransformLinear<NormalInverseGamma>? {
    r:TransformLinear<NormalInverseGamma>?;
    x1:NormalInverseGamma?;
    if !hasValue() {
      if (r <- y!.graftLinearNormalInverseGamma(compare))? {
        r!.add(z!);
      } else if (r <- z!.graftLinearNormalInverseGamma(compare))? {
        r!.add(y!);
      } else if (x1 <- y!.graftNormalInverseGamma(compare))? {
        r <- TransformLinear<NormalInverseGamma>(box(1.0), x1!, z!);
      } else if (x1 <- z!.graftNormalInverseGamma(compare))? {
        r <- TransformLinear<NormalInverseGamma>(box(1.0), x1!, y!);
      }
    }
    return r;
  }

  override function graftDotNormalInverseGamma(compare:Distribution<Real>) ->
      TransformDot<MultivariateNormalInverseGamma>? {
    r:TransformDot<MultivariateNormalInverseGamma>?;
    if !hasValue() {
      if (r <- y!.graftDotNormalInverseGamma(compare))? {
        r!.add(z!);
      } else if (r <- z!.graftDotNormalInverseGamma(compare))? {
        r!.add(y!);
      }
    }
    return r;
  }
}

/**
 * Lazy add.
 */
operator (y:Expression<Real> + z:Expression<Real>) -> Add {
  return construct<Add>(y, z);
}

/**
 * Lazy add.
 */
operator (y:Real + z:Expression<Real>) -> Add {
  return box(y) + z;
}

/**
 * Lazy add.
 */
operator (y:Expression<Real> + z:Real) -> Add {
  return y + box(z);
}
