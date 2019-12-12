/**
 * Lazy multiply.
 */
final class DiscreteMultiply<Left,Right,Value>(left:Expression<Left>,
    right:Expression<Right>) < BinaryExpression<Left,Right,Value>(left, right) {  
  function graft(child:Delay?) -> Expression<Value> {
    return left.graft(child)*right.graft(child);
  }

  function doValue(l:Left, r:Right) -> Value {
    return l*r;
  }
  
  function doGradient(d:Value, l:Left, r:Right) -> (Left, Right) {
    return (d*r, d*l);
  }

  function graftDiscrete(child:Delay?) -> DelayDiscrete? {
    y:DelayDiscrete? <- graftBoundedDiscrete(child);
    if !y? {
      x:DelayDiscrete?;
      if (x <- left.graftDiscrete(child))? {
        y <- DelayLinearDiscrete(nil, true, right.value(), x!, 0);
      } else if (x <- right.graftDiscrete(child))? {
        y <- DelayLinearDiscrete(nil, true, left.value(), x!, 0);
      }
    }
    return y;
  }

  function graftBoundedDiscrete(child:Delay?) -> DelayBoundedDiscrete? {
    y:DelayBoundedDiscrete?;
    x1:DelayBoundedDiscrete? <- left.graftBoundedDiscrete(child);
    x2:DelayBoundedDiscrete? <- right.graftBoundedDiscrete(child);

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
