/**
 * Lazy negation.
 */
final class DiscreteNegate(x:Expression<Integer>) <
    ScalarUnaryExpression<Expression<Integer>,Integer>(x) {
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

  override function graftDiscrete() -> Discrete? {
    r:Discrete? <- graftBoundedDiscrete();
    if !r? {
      x:Discrete?;
      if (x <- single!.graftDiscrete())? {
        r <- LinearDiscrete(box(-1), x!, box(0));
      }
    }
    return r;
  }

  override function graftBoundedDiscrete() -> BoundedDiscrete? {
    x:BoundedDiscrete?;
    r:BoundedDiscrete?;

    if (x <- single!.graftBoundedDiscrete())? {
      r <- LinearBoundedDiscrete(box(-1), x!, box(0));
    }
    return r;
  }
}

/**
 * Lazy negation.
 */
operator (-x:Expression<Integer>) -> Expression<Integer> {
  if x.isConstant() {
    return box(-x.value());
  } else {
    m:DiscreteNegate(x);
    return m;
  }
}
