/**
 * Lazy multiply.
 */
final class DiscreteMultiply<Left,Right,Value>(left:Left, right:Right) <
    ScalarBinaryExpression<Left,Right,Value>(left, right) {  
  override function doValue() {
    x <- left!.value()*right!.value();
  }

  override function doPilot() {
    x <- left!.pilot()*right!.pilot();
  }

  override function doMove(κ:Kernel) {
    x <- left!.move(κ)*right!.move(κ);
  }
  
  override function doGrad() {
    left!.grad(d!*right!.get());
    right!.grad(d!*left!.get());
  }

  override function graftDiscrete() -> Discrete? {
    r:Discrete?;
    if !hasValue() {
      r <- graftBoundedDiscrete();
      if !r? {
        x:Discrete?;
        if (x <- left!.graftDiscrete())? {
          r <- LinearDiscrete(right!, x!, box(0));
        } else if (x <- right!.graftDiscrete())? {
          r <- LinearDiscrete(left!, x!, box(0));
        }
      }
    }
    return r;
  }

  override function graftBoundedDiscrete() -> BoundedDiscrete? {
    r:BoundedDiscrete?;
    if !hasValue() {
      auto x1 <- left!.graftBoundedDiscrete();
      auto x2 <- right!.graftBoundedDiscrete();
      if x1? {
        r <- LinearBoundedDiscrete(right!, x1!, box(0));
      } else if x2? {
        r <- LinearBoundedDiscrete(left!, x2!, box(0));
      }
    }
    return r;
  }
}

/**
 * Lazy add.
 */
operator (left:Expression<Integer>*right:Expression<Integer>) ->
    Expression<Integer> {
  if left.isConstant() && right.isConstant() {
    return box(left.value() + right.value());
  } else {
    return construct<DiscreteMultiply<Expression<Integer>,Expression<Integer>,Integer>>(left, right);
  }
}

/**
 * Lazy add.
 */
operator (left:Integer*right:Expression<Integer>) -> Expression<Integer> {
  if right.isConstant() {
    return box(left + right.value());
  } else {
    return box(left)*right;
  }
}

/**
 * Lazy add.
 */
operator (left:Expression<Integer>*right:Integer) -> Expression<Integer> {
  if left.isConstant() {
    return box(left.value() + right);
  } else {
    return left*box(right);
  }
}
