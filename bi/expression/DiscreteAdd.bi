/**
 * Lazy add.
 */
final class DiscreteAdd<Left,Right,Value>(left:Expression<Left>,
    right:Expression<Right>) < BinaryExpression<Left,Right,Value>(left, right) {  
  override function computeValue(l:Left, r:Right) -> Value {
    return l + r;
  }
  
  override function computeGrad(d:Value, l:Left, r:Right) -> (Left, Right) {
    return (d, d);
  }

  override function graftDiscrete() -> Discrete? {
    r:Discrete? <- graftBoundedDiscrete();
    if !r? {
      /* match a template */
      x:Discrete?;
      if (x <- left.graftDiscrete())? {
        r <- LinearDiscrete(Boxed(1), x!, right);
      } else if (x <- right.graftDiscrete())? {
        r <- LinearDiscrete(Boxed(1), x!, left);
      }
    }
    return r;
  }

  override function graftBoundedDiscrete() -> BoundedDiscrete? {
    x1:BoundedDiscrete? <- left.graftBoundedDiscrete();
    x2:BoundedDiscrete? <- right.graftBoundedDiscrete();
    r:BoundedDiscrete?;

    /* match a template */
    if x1? && x2? {
      r <- AddBoundedDiscrete(x1!, x2!);
    } else if x1? {
      r <- LinearBoundedDiscrete(Boxed(1), x1!, right);
    } else if x2? {
      r <- LinearBoundedDiscrete(Boxed(1), x2!, left);
    }
    return r;
  }
}

/**
 * Lazy add.
 */
operator (left:Expression<Integer> + right:Expression<Integer>) ->
    DiscreteAdd<Integer,Integer,Integer> {
  m:DiscreteAdd<Integer,Integer,Integer>(left, right);
  return m;
}

/**
 * Lazy add.
 */
operator (left:Integer + right:Expression<Integer>) ->
    DiscreteAdd<Integer,Integer,Integer> {
  return Boxed(left) + right;
}

/**
 * Lazy add.
 */
operator (left:Expression<Integer> + right:Integer) ->
    DiscreteAdd<Integer,Integer,Integer> {
  return left + Boxed(right);
}
