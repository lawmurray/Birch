/**
 * Lazy multiply.
 */
final class DiscreteMultiply<Left,Right,Value>(left:Expression<Left>,
    right:Expression<Right>) < BinaryExpression<Left,Right,Value>(left, right) {  
  function graft() -> Expression<Value> {
    return left.graft()*right.graft();
  }

  function doValue(l:Left, r:Right) -> Value {
    return l*r;
  }
  
  function doGradient(d:Value, l:Left, r:Right) -> (Left, Right) {
    return (d*r, d*l);
  }

  function graftDiscrete() -> DelayDiscrete? {
    y:DelayDiscrete? <- graftBoundedDiscrete();
    if !y? {
      x:DelayDiscrete?;
      if (x <- left.graftDiscrete())? {
        y <- DelayLinearDiscrete(nil, true, right.value(), x!, 0);
      } else if (x <- right.graftDiscrete())? {
        y <- DelayLinearDiscrete(nil, true, left.value(), x!, 0);
      }
    }
    return y;
  }

  function graftBoundedDiscrete() -> DelayBoundedDiscrete? {
    y:DelayBoundedDiscrete?;
    x1:DelayBoundedDiscrete? <- left.graftBoundedDiscrete();
    x2:DelayBoundedDiscrete? <- right.graftBoundedDiscrete();

    if x1? && !(x1!.hasValue()) {
      y <- DelayLinearBoundedDiscrete(nil, true, right.value(), x1!, 0);
    } else if x2? && !(x2!.hasValue()) {
      y <- DelayLinearBoundedDiscrete(nil, true, left.value(), x2!, 0);
    }
    return y;
  }
}

/**
 * Lazy add.
 */
operator (left:Expression<Integer>*right:Expression<Integer>) ->
    DiscreteMultiply<Integer,Integer,Integer> {
  m:DiscreteMultiply<Integer,Integer,Integer>(left, right);
  return m;
}

/**
 * Lazy add.
 */
operator (left:Integer*right:Expression<Integer>) ->
    DiscreteMultiply<Integer,Integer,Integer> {
  return Boxed(left)*right;
}

/**
 * Lazy add.
 */
operator (left:Expression<Integer>*right:Integer) ->
    DiscreteMultiply<Integer,Integer,Integer> {
  return left*Boxed(right);
}
