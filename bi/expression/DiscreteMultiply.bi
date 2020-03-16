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
    r:Discrete? <- graftBoundedDiscrete();
    if !r? {
      /* match a template */
      x:Discrete?;
      if (x <- left.graftDiscrete())? {
        r <- LinearDiscrete(right, x!, Boxed(0));
      } else if (x <- right.graftDiscrete())? {
        r <- LinearDiscrete(left, x!, Boxed(0));
      }

      /* finalize, and if not valid, return nil */
      if !r? || !r!.graftFinalize() {
        r <- nil;
      }
    }
    return r;
  }

  function graftBoundedDiscrete() -> BoundedDiscrete? {
    x1:BoundedDiscrete? <- left.graftBoundedDiscrete();
    x2:BoundedDiscrete? <- right.graftBoundedDiscrete();
    r:BoundedDiscrete?;

    /* match a template */       
    if x1? {
      r <- LinearBoundedDiscrete(right, x1!, Boxed(0));
    } else if x2? {
      r <- LinearBoundedDiscrete(left, x2!, Boxed(0));
    }
    
    /* finalize, and if not valid, return nil */
    if !r? || !r!.graftFinalize() {
      r <- nil;
    }
    return r;
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
