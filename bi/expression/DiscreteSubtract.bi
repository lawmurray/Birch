/**
 * Lazy subtract.
 */
final class DiscreteSubtract<Left,Right,Value>(left:Left, right:Right) <
    ScalarBinaryExpression<Left,Right,Value>(left, right) {  
  override function doValue() {
    x <- left.value() - right.value();
  }

  override function doPilot() {
    x <- left.pilot() - right.pilot();
  }

  override function doMove(κ:Kernel) {
    x <- left.move(κ) - right.move(κ);
  }
  
  override function doGrad() {
    left.grad(d!);
    right.grad(-d!);
  }

  override function graftDiscrete() -> Discrete? {
    r:Discrete? <- graftBoundedDiscrete();
    if !r? {
      x:Discrete?;
      if (x <- left.graftDiscrete())? {
        r <- LinearDiscrete(box(1), x!, -right);
      } else if (x <- right.graftDiscrete())? {
        r <- LinearDiscrete(box(-1), x!, left);
      }
    }
    return r;
  }

  override function graftBoundedDiscrete() -> BoundedDiscrete? {
    x1:BoundedDiscrete? <- left.graftBoundedDiscrete();
    x2:BoundedDiscrete? <- right.graftBoundedDiscrete();
    r:BoundedDiscrete?;

    if x1? && x2? {
      r <- SubtractBoundedDiscrete(x1!, x2!);
    } else if x1? {
      r <- LinearBoundedDiscrete(box(1), x1!, -right);
    } else if x2? {
      r <- LinearBoundedDiscrete(box(-1), x2!, left);
    }

    return r;
  }
}

/**
 * Lazy subtract.
 */
operator (left:Expression<Integer> - right:Expression<Integer>) ->
    Expression<Integer> {
  if left.isConstant() && right.isConstant() {
    return box(left.value() - right.value());
  } else {
    m:DiscreteSubtract<Expression<Integer>,Expression<Integer>,Integer>(left, right);
    return m;
  }
}

/**
 * Lazy subtract.
 */
operator (left:Integer - right:Expression<Integer>) -> Expression<Integer> {
  if right.isConstant() {
    return box(left - right.value());
  } else {
    return box(left) - right;
  }
}

/**
 * Lazy subtract.
 */
operator (left:Expression<Integer> - right:Integer) -> Expression<Integer> {
  if left.isConstant() {
    return box(left.value() + right);
  } else {
    return left - box(right);
  }
}
