/**
 * Lazy multiply.
 */
final class Multiply<Left,Right,Value>(left:Left, right:Right) <
    ScalarBinaryExpression<Left,Right,Value>(left, right) {  
  override function doValue() {
    x <- left.value()*right.value();
  }

  override function doPilot() {
    x <- left.pilot()*right.pilot();
  }

  override function doGet() {
    x <- left.get()*right.get();
  }

  override function doMove(κ:Kernel) {
    x <- left.move(κ)*right.move(κ);
  }

  override function doGrad() {
    left.grad(d!*right.get());
    right.grad(d!*left.get());
  }

  override function graftScaledGamma() -> TransformLinear<Gamma>? {
    y:TransformLinear<Gamma>?;
    z:Gamma?;
    
    if (y <- left.graftScaledGamma())? {
      y!.multiply(right);
    } else if (y <- right.graftScaledGamma())? {
      y!.multiply(left);
    } else if (z <- left.graftGamma())? {
      y <- TransformLinear<Gamma>(right, z!);
    } else if (z <- right.graftGamma())? {
      y <- TransformLinear<Gamma>(left, z!);
    }
    return y;
  }

  override function graftLinearGaussian() -> TransformLinear<Gaussian>? {
    y:TransformLinear<Gaussian>?;
    z:Gaussian?;
    
    if (y <- left.graftLinearGaussian())? {
      y!.multiply(right);
    } else if (y <- right.graftLinearGaussian())? {
      y!.multiply(left);
    } else if (z <- left.graftGaussian())? {
      y <- TransformLinear<Gaussian>(right, z!);
    } else if (z <- right.graftGaussian())? {
      y <- TransformLinear<Gaussian>(left, z!);
    }
    return y;
  }

  override function graftDotGaussian() -> TransformDot<MultivariateGaussian>? {
    y:TransformDot<MultivariateGaussian>?;
    
    if (y <- left.graftDotGaussian())? {
      y!.multiply(right);
    } else if (y <- right.graftDotGaussian())? {
      y!.multiply(left);
    }
    return y;
  }
 
  override function graftLinearNormalInverseGamma(compare:Distribution<Real>) ->
      TransformLinear<NormalInverseGamma>? {
    y:TransformLinear<NormalInverseGamma>?;
    z:NormalInverseGamma?;
    
    if (y <- left.graftLinearNormalInverseGamma(compare))? {
      y!.multiply(right);
    } else if (y <- right.graftLinearNormalInverseGamma(compare))? {
      y!.multiply(left);
    } else if (z <- left.graftNormalInverseGamma(compare))? {
      y <- TransformLinear<NormalInverseGamma>(right, z!);
    } else if (z <- right.graftNormalInverseGamma(compare))? {
      y <- TransformLinear<NormalInverseGamma>(left, z!);
    }
    return y;
  }

  override function graftDotNormalInverseGamma(compare:Distribution<Real>) ->
      TransformDot<MultivariateNormalInverseGamma>? {
    y:TransformDot<MultivariateNormalInverseGamma>?;
    
    if (y <- left.graftDotNormalInverseGamma(compare))? {
      y!.multiply(right);
    } else if (y <- right.graftDotNormalInverseGamma(compare))? {
      y!.multiply(left);
    }
    return y;
  }
}

/**
 * Lazy multiply.
 */
operator (left:Expression<Real>*right:Expression<Real>) -> Expression<Real> {
  if left.isConstant() && right.isConstant() {
    return box(left.value()*right.value());
  } else {
    m:Multiply<Expression<Real>,Expression<Real>,Real>(left, right);
    return m;
  }
}

/**
 * Lazy multiply.
 */
operator (left:Real*right:Expression<Real>) -> Expression<Real> {
  if right.isConstant() {
    return box(left*right.value());
  } else {
    return box(left)*right;
  }
}

/**
 * Lazy multiply.
 */
operator (left:Expression<Real>*right:Real) -> Expression<Real> {
  if left.isConstant() {
    return box(left.value()*right);
  } else {
    return left*box(right);
  }
}
