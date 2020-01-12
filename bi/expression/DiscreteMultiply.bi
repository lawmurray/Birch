/**
 * Lazy multiply.
 */
final class DiscreteMultiply<Left,Right,Value>(left:Expression<Left>,
    right:Expression<Right>) < BinaryExpression<Left,Right,Value>(left, right) {  
  function doValue(l:Left, r:Right) -> Value {
    return l*r;
  }
  
  function doGradient(d:Value, l:Left, r:Right) -> (Left, Right) {
    return (d*r, d*l);
  }

  function graftDiscrete() -> Discrete? {
    y:Discrete? <- graftBoundedDiscrete();
    if !y? {
      x:Discrete?;
      if (x <- left.graftDiscrete())? {
        y <- LinearDiscrete(right.value(), x!, 0);
      } else if (x <- right.graftDiscrete())? {
        y <- LinearDiscrete(left.value(), x!, 0);
      }
    }
    return y;
  }

  function graftBoundedDiscrete() -> BoundedDiscrete? {
    y:BoundedDiscrete?;
    x1:BoundedDiscrete? <- left.graftBoundedDiscrete();
    x2:BoundedDiscrete? <- right.graftBoundedDiscrete();

    if x1? && !(x1!.hasValue()) {
      y <- LinearBoundedDiscrete(right.value(), x1!, 0);
    } else if x2? && !(x2!.hasValue()) {
      y <- LinearBoundedDiscrete(left.value(), x2!, 0);
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
