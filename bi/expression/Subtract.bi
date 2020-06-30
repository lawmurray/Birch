/**
 * Lazy subtract.
 */
final class Subtract<Left,Right,Value>(left:Left, right:Right) <
    ScalarBinaryExpression<Left,Right,Value>(left, right) {
  override function doValue() {
    x <- left!.value() - right!.value();
  }

  override function doPilot() {
    x <- left!.pilot() - right!.pilot();
  }

  override function doMove(κ:Kernel) {
    x <- left!.move(κ) - right!.move(κ);
  }
  
  override function doGrad() {
    left!.grad(d!);
    right!.grad(-d!);
  }

  override function graftLinearGaussian() -> TransformLinear<Gaussian>? {
    y:TransformLinear<Gaussian>?;
    if !hasValue() {
      z:Gaussian?;
    
      if (y <- left!.graftLinearGaussian())? {
        y!.add(-right!);
      } else if (y <- right!.graftLinearGaussian())? {
        y!.negateAndAdd(left!);
      } else if (z <- left!.graftGaussian())? {
        y <- TransformLinear<Gaussian>(box(1.0), z!, -right!);
      } else if (z <- right!.graftGaussian())? {
        y <- TransformLinear<Gaussian>(box(-1.0), z!, left!);
      }
    }
    return y;
  }

  override function graftDotGaussian() -> TransformDot<MultivariateGaussian>? {
    y:TransformDot<MultivariateGaussian>?;
    if !hasValue() {
      z:Gaussian?;
    
      if (y <- left!.graftDotGaussian())? {
        y!.add(-right!);
      } else if (y <- right!.graftDotGaussian())? {
        y!.negateAndAdd(left!);
      }
    }
    return y;
  }

  override function graftLinearNormalInverseGamma(compare:Distribution<Real>) ->
      TransformLinear<NormalInverseGamma>? {
    y:TransformLinear<NormalInverseGamma>?;
    if !hasValue() {
      z:NormalInverseGamma?;

      if (y <- left!.graftLinearNormalInverseGamma(compare))? {
        y!.subtract(right!);
      } else if (y <- right!.graftLinearNormalInverseGamma(compare))? {
        y!.negateAndAdd(left!);
      } else if (z <- left!.graftNormalInverseGamma(compare))? {
        y <- TransformLinear<NormalInverseGamma>(box(1.0), z!, -right!);
      } else if (z <- right!.graftNormalInverseGamma(compare))? {
        y <- TransformLinear<NormalInverseGamma>(box(-1.0), z!, left!);
      }
    }
    return y;
  }

  override function graftDotNormalInverseGamma(compare:Distribution<Real>) ->
      TransformDot<MultivariateNormalInverseGamma>? {
    y:TransformDot<MultivariateNormalInverseGamma>?;
    if !hasValue() {
      if (y <- left!.graftDotNormalInverseGamma(compare))? {
        y!.subtract(right!);
      } else if (y <- right!.graftDotNormalInverseGamma(compare))? {
        y!.negateAndAdd(left!);
      }
    }
    return y;
  }
}

/**
 * Lazy subtract.
 */
operator (left:Expression<Real> - right:Expression<Real>) ->
    Expression<Real> {
  if left.isConstant() && right.isConstant() {
    return box(left.value() - right.value());
  } else {
    m:Subtract<Expression<Real>,Expression<Real>,Real>(left, right);
    return m;
  }
}

/**
 * Lazy subtract.
 */
operator (left:Real - right:Expression<Real>) -> Expression<Real> {
  if right.isConstant() {
    return box(left - right.value());
  } else {
    return box(left) - right;
  }
}

/**
 * Lazy subtract.
 */
operator (left:Expression<Real> - right:Real) -> Expression<Real> {
  if left.isConstant() {
    return box(left.value() - right);
  } else {
    return left - box(right);
  }
}
