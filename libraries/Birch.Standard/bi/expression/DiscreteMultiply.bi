/**
 * Lazy multiply.
 */
final class DiscreteMultiply(y:Expression<Integer>, z:Expression<Integer>) <
    ScalarBinaryExpression<Expression<Integer>,Expression<Integer>,Integer,
    Integer,Real,Real,Integer>(y, z) {  
  override function doEvaluate(y:Integer, z:Integer) -> Integer {
    return y*z;
  }

  override function doEvaluateGradLeft(d:Real, x:Integer, y:Integer,
      z:Integer) -> Real {
    return d*z;
  }

  override function doEvaluateGradRight(d:Real, x:Integer, y:Integer,
      z:Integer) -> Real {
    return d*y;
  }

  override function graftDiscrete() -> Discrete? {
    r:Discrete?;
    if !hasValue() {
      r <- graftBoundedDiscrete();
      if !r? {
        x1:Discrete?;
        if (x1 <- y!.graftDiscrete())? {
          r <- LinearDiscrete(z!, x1!, box(0));
        } else if (x1 <- z!.graftDiscrete())? {
          r <- LinearDiscrete(y!, x1!, box(0));
        }
      }
    }
    return r;
  }

  override function graftBoundedDiscrete() -> BoundedDiscrete? {
    r:BoundedDiscrete?;
    if !hasValue() {
      auto x1 <- y!.graftBoundedDiscrete();
      auto x2 <- z!.graftBoundedDiscrete();
      if x1? {
        r <- LinearBoundedDiscrete(z!, x1!, box(0));
      } else if x2? {
        r <- LinearBoundedDiscrete(y!, x2!, box(0));
      }
    }
    return r;
  }
}

/**
 * Lazy multiply.
 */
operator (y:Expression<Integer>*z:Expression<Integer>) -> DiscreteMultiply {
  return construct<DiscreteMultiply>(y, z);
}

/**
 * Lazy multiply.
 */
operator (y:Integer*z:Expression<Integer>) -> DiscreteMultiply {
  return box(y)*z;
}

/**
 * Lazy multiply.
 */
operator (y:Expression<Integer>*z:Integer) -> DiscreteMultiply {
  return y*box(z);
}
